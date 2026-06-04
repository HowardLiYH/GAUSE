"""
V4 EG update tests.

These tests verify the structural correctness of the V4 canonical
(exponentiated-gradient) affinity update, and document the contrast
with the V3 legacy additive update.

The tests are intentionally focused on the *mechanics* of the update
rule, not on emergent specialization dynamics. Specialization behavior
is covered by the experiment scripts under ``experiments/``.

Properties verified for V4 (EG):
    - Simplex sum stays at exactly 1.0 (within float epsilon)
    - All entries remain strictly positive after any update
    - The V3 clamp path is never invoked
    - Winning regime's affinity increases after a win
    - Non-winning regimes' affinities decrease by the same proportional factor
    - Update is order-invariant (same multiset of wins -> same final state)

Properties verified for V3 (legacy):
    - The undocumented max(0.01, ...) clamp eventually fires for plausible inputs
    - The pre-normalization sum drifts below 1 after winning rounds

Cross-version property:
    - At alpha = 1/R (uniform initial state), V3 and V4 first-step
      updates differ by O(eta^2) (first-order equivalent)
"""

import math
from typing import Dict, List

import pytest

from src.agents.niche_population import NicheAgent, NichePopulation


REGIMES = ["regime_0", "regime_1", "regime_2", "regime_3"]
METHODS = ["m_naive", "m_trend", "m_mean_revert"]


def _make_agent(update_rule: str, eta: float = 0.1, seed: int = 42) -> NicheAgent:
    return NicheAgent(
        agent_id="test_agent",
        regimes=REGIMES,
        seed=seed,
        methods=METHODS,
        learning_rate=eta,
        update_rule=update_rule,
    )


def _affinity_vector(agent: NicheAgent) -> List[float]:
    """Return affinity values in canonical regime order."""
    return [agent.niche_affinity[r] for r in REGIMES]


# =====================================================================
# V4 (EG) structural properties
# =====================================================================


class TestEGSimplexPreservation:
    """V4 EG must preserve the probability simplex exactly."""

    def test_initial_state_on_simplex(self):
        agent = _make_agent("eg")
        v = _affinity_vector(agent)
        assert sum(v) == pytest.approx(1.0, abs=1e-12)
        assert all(x > 0 for x in v)

    def test_single_update_on_simplex(self):
        agent = _make_agent("eg")
        agent._update_niche_affinity_eg("regime_0")
        v = _affinity_vector(agent)
        assert sum(v) == pytest.approx(1.0, abs=1e-12)
        assert all(x > 0 for x in v)

    def test_many_updates_on_simplex(self):
        agent = _make_agent("eg")
        for _ in range(200):
            agent._update_niche_affinity_eg("regime_0")
        v = _affinity_vector(agent)
        assert sum(v) == pytest.approx(1.0, abs=1e-9)
        assert all(x > 0 for x in v)

    def test_mixed_wins_on_simplex(self):
        agent = _make_agent("eg")
        win_sequence = ["regime_0", "regime_1", "regime_2", "regime_0", "regime_3"] * 20
        for r in win_sequence:
            agent._update_niche_affinity_eg(r)
        v = _affinity_vector(agent)
        assert sum(v) == pytest.approx(1.0, abs=1e-9)
        assert all(x > 0 for x in v)


class TestEGNonNegativity:
    """V4 EG must never produce zero or negative entries."""

    def test_concentrated_winning(self):
        """Even after 1000 wins on the same regime, others stay positive."""
        agent = _make_agent("eg")
        for _ in range(1000):
            agent._update_niche_affinity_eg("regime_0")
        v = _affinity_vector(agent)
        # Other regimes get exponentially small but never zero
        for x in v:
            assert x > 0, f"Found non-positive entry: {x}"
        # The winning regime should dominate
        assert v[0] > 0.99

    def test_no_clamp_invocations(self):
        """The V3 clamp diagnostic counter must stay at 0 under EG."""
        agent = _make_agent("eg")
        for _ in range(500):
            agent._update_niche_affinity_eg("regime_0")
        assert agent._diag_clamp_invocations == 0


class TestEGMonotonicity:
    """V4 EG must move affinity in the expected direction."""

    def test_winner_grows(self):
        agent = _make_agent("eg")
        before = agent.niche_affinity["regime_0"]
        agent._update_niche_affinity_eg("regime_0")
        after = agent.niche_affinity["regime_0"]
        assert after > before

    def test_others_shrink(self):
        agent = _make_agent("eg")
        before = {r: agent.niche_affinity[r] for r in REGIMES if r != "regime_0"}
        agent._update_niche_affinity_eg("regime_0")
        after = {r: agent.niche_affinity[r] for r in REGIMES if r != "regime_0"}
        for r in before:
            assert after[r] < before[r], f"{r}: {before[r]} -> {after[r]} did not shrink"

    def test_proportional_shrinkage(self):
        """
        EG shrinks all non-winners by the *same factor* (the normalization
        divisor). The ratio between any two non-winning entries must be
        preserved exactly across the update.
        """
        agent = _make_agent("eg")
        # Slightly perturb initial state so ratios are non-trivial
        agent.niche_affinity = {
            "regime_0": 0.4,
            "regime_1": 0.3,
            "regime_2": 0.2,
            "regime_3": 0.1,
        }
        ratio_before = (
            agent.niche_affinity["regime_1"] / agent.niche_affinity["regime_2"]
        )
        agent._update_niche_affinity_eg("regime_0")
        ratio_after = (
            agent.niche_affinity["regime_1"] / agent.niche_affinity["regime_2"]
        )
        assert ratio_after == pytest.approx(ratio_before, rel=1e-9)


class TestEGOrderInvariance:
    """Same multiset of wins -> same final state (up to float order)."""

    def test_order_invariant(self):
        wins_a = ["regime_0", "regime_1", "regime_0", "regime_2", "regime_1"]
        wins_b = list(reversed(wins_a))

        agent_a = _make_agent("eg", seed=7)
        agent_b = _make_agent("eg", seed=7)

        for r in wins_a:
            agent_a._update_niche_affinity_eg(r)
        for r in wins_b:
            agent_b._update_niche_affinity_eg(r)

        for r in REGIMES:
            assert agent_a.niche_affinity[r] == pytest.approx(
                agent_b.niche_affinity[r], abs=1e-9
            )


# =====================================================================
# V3 (legacy) structural issues (regression-style: should reproduce bugs)
# =====================================================================


class TestV3StructuralIssues:
    """Document the structural problems with the V3 update rule."""

    def test_v3_clamp_eventually_fires(self):
        """
        In a sufficiently long V3 run with concentrated wins, the
        max(0.01, ...) clamp must fire. This is the 'undocumented duct
        tape' that the published main paper does not mention.
        """
        agent = _make_agent("v3_additive")
        for _ in range(200):
            agent._update_niche_affinity_v3("regime_0")
        assert agent._diag_clamp_invocations > 0, (
            "Expected V3 clamp to fire under concentrated wins; if this "
            "test passes without clamping, V3's analytical issue may be "
            "less severe than predicted."
        )

    def test_v3_premass_drifts_below_one(self):
        """
        Per the audit report, the V3 update's pre-normalization sum drifts
        away from 1.0 each win. Specifically, the drift is approximately
        ``-eta * alpha_winner``, which is in [-eta, 0).

        We verify the sum drops below 1.0 by a meaningful margin.
        """
        agent = _make_agent("v3_additive", eta=0.1)
        # Take one update from uniform state
        agent._update_niche_affinity_v3("regime_0")
        # alpha_winner before update was 0.25, so expected drift ~ -0.025
        last_sum = agent._diag_premass_sum_history[-1]
        assert last_sum < 1.0
        assert last_sum < 0.99  # well outside numerical noise


# =====================================================================
# Cross-version: first-order step-size gap at uniform start
# =====================================================================


class TestEGV3FirstOrderStepSizeGap:
    """
    The V3 and V4 updates are NOT first-order equivalent at uniform
    start. Working out the math at alpha = 1/R, the per-step gain on the
    winning entry to leading order in eta is:

        V3:  delta_alpha_winner  =  eta * (1 - 1/R + 1/R^2)
        V4:  delta_alpha_winner  =  eta * (1/R) * (1 - 1/R)

        Ratio V3/V4  =  (R^2 - R + 1) / (R - 1)

    For R = 4, the ratio is 13/3 ~= 4.33. That is, V3 with eta = 0.1
    moves the winning entry per step about 4.33x as much as V4 with the
    same eta. This means re-using eta = 0.1 in V4 will produce noticeably
    slower specialization dynamics than V3 at R = 4.

    To match V3's empirical specialization timescale in V4, one should
    use eta_v4 ~= 4.33 * eta_v3 at R = 4 (more generally,
    (R^2 - R + 1) / (R - 1) * eta_v3).

    This test verifies the analytical ratio holds numerically.
    """

    def _v3_first_order_winner_gain(self, eta: float, R: int) -> float:
        """Closed-form V3 winner gain at uniform start, leading order in eta."""
        return eta * (1 - 1 / R + 1 / R**2)

    def _v4_first_order_winner_gain(self, eta: float, R: int) -> float:
        """Closed-form V4 winner gain at uniform start, leading order in eta."""
        return eta * (1 / R) * (1 - 1 / R)

    def test_v3_winner_gain_matches_closed_form(self):
        eta = 0.01
        R = len(REGIMES)
        agent_v3 = _make_agent("v3_additive", eta=eta)
        before = agent_v3.niche_affinity["regime_0"]
        agent_v3._update_niche_affinity_v3("regime_0")
        after = agent_v3.niche_affinity["regime_0"]
        observed = after - before
        expected = self._v3_first_order_winner_gain(eta, R)
        # Numerical leading-order match; allow O(eta^2) slack
        assert abs(observed - expected) < eta**2 * 10, (
            f"V3 winner gain {observed} vs expected {expected}"
        )

    def test_v4_winner_gain_matches_closed_form(self):
        eta = 0.01
        R = len(REGIMES)
        agent_eg = _make_agent("eg", eta=eta)
        before = agent_eg.niche_affinity["regime_0"]
        agent_eg._update_niche_affinity_eg("regime_0")
        after = agent_eg.niche_affinity["regime_0"]
        observed = after - before
        expected = self._v4_first_order_winner_gain(eta, R)
        assert abs(observed - expected) < eta**2 * 10, (
            f"V4 winner gain {observed} vs expected {expected}"
        )

    def test_v3_v4_first_order_ratio(self):
        """
        Confirm the per-step gain ratio is (R^2 - R + 1) / (R - 1) at
        uniform start. This is the empirical anchor for the V4 eta
        rescaling recommendation in the audit report.
        """
        eta = 0.01
        R = len(REGIMES)
        agent_v3 = _make_agent("v3_additive", eta=eta)
        agent_eg = _make_agent("eg", eta=eta)
        v3_before = agent_v3.niche_affinity["regime_0"]
        eg_before = agent_eg.niche_affinity["regime_0"]
        agent_v3._update_niche_affinity_v3("regime_0")
        agent_eg._update_niche_affinity_eg("regime_0")
        v3_gain = agent_v3.niche_affinity["regime_0"] - v3_before
        eg_gain = agent_eg.niche_affinity["regime_0"] - eg_before
        observed_ratio = v3_gain / eg_gain
        expected_ratio = (R**2 - R + 1) / (R - 1)
        # Leading-order ratio; tolerate O(eta) correction
        assert abs(observed_ratio - expected_ratio) < 0.5, (
            f"V3/V4 ratio {observed_ratio:.4f} vs expected {expected_ratio:.4f}"
        )

    def test_eta_rescaling_recipe_works(self):
        """
        Verify that V4 with eta_v4 = (R^2-R+1)/(R-1) * eta_v3 reproduces
        V3's first-order step size at uniform start. This is the
        recommended recipe for 'V4 with matched dynamics'.
        """
        eta_v3 = 0.01
        R = len(REGIMES)
        scale = (R**2 - R + 1) / (R - 1)
        eta_v4 = scale * eta_v3
        agent_v3 = _make_agent("v3_additive", eta=eta_v3)
        agent_eg = _make_agent("eg", eta=eta_v4)
        v3_before = agent_v3.niche_affinity["regime_0"]
        eg_before = agent_eg.niche_affinity["regime_0"]
        agent_v3._update_niche_affinity_v3("regime_0")
        agent_eg._update_niche_affinity_eg("regime_0")
        v3_gain = agent_v3.niche_affinity["regime_0"] - v3_before
        eg_gain = agent_eg.niche_affinity["regime_0"] - eg_before
        # After rescaling, gains should match to O(eta^2)
        assert abs(v3_gain - eg_gain) < eta_v3**2 * 10, (
            f"After rescaling, V3 gain {v3_gain:.6f} vs V4 gain {eg_gain:.6f}"
        )


# =====================================================================
# Population-level wiring tests
# =====================================================================


class TestPopulationWiring:
    """Make sure the update_rule propagates from Population to agents."""

    def test_population_threads_update_rule(self):
        pop = NichePopulation(
            n_agents=3,
            regimes=REGIMES,
            seed=42,
            update_rule="eg",
        )
        for agent in pop.agents.values():
            assert agent.update_rule == "eg"

    def test_population_v3_mode(self):
        pop = NichePopulation(
            n_agents=3,
            regimes=REGIMES,
            seed=42,
            update_rule="v3_additive",
        )
        for agent in pop.agents.values():
            assert agent.update_rule == "v3_additive"

    def test_population_rejects_bad_rule(self):
        with pytest.raises(ValueError, match="update_rule must be one of"):
            NichePopulation(
                n_agents=3,
                regimes=REGIMES,
                seed=42,
                update_rule="not_a_real_rule",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
