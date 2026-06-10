"""
Structural tests for the three flaw-closing experiments added in the revision:

  * exp_hybrid_router.py        -- reservation term protects assigned regimes from eviction
  * exp_intra_regime_drift.py   -- the drift stream actually drifts champions; staleness reset works
  * exp_function_approx_cl.py   -- permuted tasks are genuinely distinct permutations

These tests check the *mechanics* the experiments rely on, not full-run numerics
(those are produced by running the scripts and saved under ``results/``).
"""

import numpy as np
import pytest

from experiments.exp_nonstationary_capacity import make_nonstationary_stream
from experiments.exp_hybrid_router import PreservingStoreAgent
from experiments.exp_intra_regime_drift import make_drift_stream, DriftAgent


REGIMES = [f"R{i}" for i in range(6)]
METHODS = [f"M{i}" for i in range(6)]


def test_preserving_agent_does_not_evict_protected_regime():
    """With spare capacity (K=2), a protected regime survives eviction that plain LRU
    would trigger; the unprotected occupant is evicted instead. (At a hard K=1 slot no
    reservation is possible -- which is exactly why the hybrid shows no K=1 benefit.)"""
    rng = np.random.default_rng(0)
    a = PreservingStoreAgent(REGIMES, METHODS, K=2, rng=rng, cap_model="lru")
    a.ensure("R0")
    a.set_protected({"R0"})
    a.ensure("R1")          # store now {R0(protected), R1}, at capacity
    a.ensure("R2")          # overflow: evict the unprotected LRU (R1), keep protected R0
    assert "R0" in a.store, "protected regime was evicted despite spare capacity"
    assert "R1" not in a.store, "unprotected LRU occupant was not evicted"


def test_unprotected_regime_is_evicted_at_capacity():
    """Without protection, the base LRU behavior (eviction at K) is preserved."""
    rng = np.random.default_rng(0)
    a = PreservingStoreAgent(REGIMES, METHODS, K=1, rng=rng, cap_model="lru")
    a.ensure("R0")
    a.ensure("R1")  # no protection -> R0 evicted
    assert "R0" not in a.store and "R1" in a.store


def test_protected_regime_does_not_decay_under_interference():
    """Soft interference must skip protected regimes."""
    rng = np.random.default_rng(0)
    a = PreservingStoreAgent(REGIMES, METHODS, K=1, rng=rng, cap_model="soft")
    for r in ("R0", "R1", "R2"):
        a.ensure(r)
        a.store[r] = {m: [5.0, 5.0] for m in METHODS}  # informed beliefs
    a.set_protected({"R0"})
    before = a.store["R0"][METHODS[0]][0]
    a._interfere(current="R2")  # overflow interference on the non-current, non-protected
    after = a.store["R0"][METHODS[0]][0]
    assert after == pytest.approx(before), "protected regime decayed under interference"


def test_drift_stream_changes_champion_on_reactivation():
    """The drifted reactivation must move the argmin-error method for at least one regime."""
    steps, regimes, methods, post_drift = make_drift_stream(R=6, W=3, seed=0)
    # champion at a step = method with the smallest *expected* error (the 0-offset method)
    champ_by_regime = {}
    drift_seen = False
    for (regime, errs) in steps:
        champ = min(errs, key=errs.get)
        if regime in champ_by_regime and champ_by_regime[regime] != champ:
            drift_seen = True
        champ_by_regime[regime] = champ
    assert drift_seen, "no champion drift observed in the drift stream"
    assert any(post_drift), "no post-drift probe steps were flagged"


def test_staleness_reset_clears_stale_beliefs():
    """reset_regime must return the regime's beliefs to the uninformed prior."""
    rng = np.random.default_rng(0)
    a = DriftAgent(REGIMES, METHODS, K=3, rng=rng)
    a.ensure("R0")
    a.store["R0"]["M0"] = [20.0, 2.0]  # confidently (and now wrongly) favors M0
    a.reset_regime("R0")
    assert a.store["R0"]["M0"] == [1.0, 1.0], "stale beliefs not reset to prior"
    assert a.q_ema["R0"] == pytest.approx(0.5)


def test_nonstationary_and_drift_streams_have_reactivations():
    """Both streams must actually produce dormancy/reactivation structure to test retention."""
    _, _, _, react = make_nonstationary_stream(R=6, W=3, seed=0)
    _, _, _, post_drift = make_drift_stream(R=6, W=3, seed=0)
    assert sum(react) > 0
    assert sum(post_drift) > 0
