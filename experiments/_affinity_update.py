"""
Shared affinity-update helpers used by the experiment scripts.

The experiment scripts (e.g. ``exp_unified_pipeline.py``) re-implement the
NichePopulation competitive dynamics inline rather than instantiating
``src.agents.niche_population.NichePopulation``. They do so for two reasons:

  1. The experiment scripts predate the consolidated ``NicheAgent`` class
     and were the canonical inputs to the published numbers.
  2. They use a flat dict-of-dicts representation for affinities, which is
     more amenable to per-experiment customization than the OOP API.

To avoid duplicating the V3 vs V4 update logic across seven experiment
files, the two update rules live here as small pure functions and are
called by name via :func:`apply_affinity_update`.

API
---
- :func:`apply_affinity_update(affinity, winning_regime, regimes, eta, rule)`:
    Returns a new affinity dict for the winner. ``rule`` is one of
    ``"eg"`` (V4 canonical) or ``"v3_additive"`` (V3 legacy).

Notes
-----
The V4 path is the default for all experiments. V3 remains available via
``update_rule="v3_additive"`` solely for direct comparison runs, and is
preserved verbatim from the published v1.0--v3.x implementation.
"""

from __future__ import annotations

import math
from typing import Dict, List


def eg_eta_for_regimes(n_regimes: int, base_eta: float = 0.1) -> float:
    """Return the EG learning rate rescaled to match V3's first-order step.

    The exponentiated-gradient (V4) update takes smaller steps than the V3
    additive update at the same nominal ``base_eta``. At a uniform start with
    ``R`` regimes, V3 moves the winning affinity ``(R^2 - R + 1) / (R - 1)``
    times as far per win as V4 (the ``(1 - alpha)`` paper-V3 variant). To make
    V4 reproduce V3's specialization timescale over the same iteration budget,
    we scale eta up by that factor::

        eta_V4(R) = base_eta * (R^2 - R + 1) / (R - 1)

    This is exactly the rescale applied by the canonical headline pipeline
    (``exp_unified_pipeline.lr_for_domain``) and ``exp_method_specialization``.
    Using it in every EG experiment keeps the auxiliary scripts numerically
    consistent with the published tables. See
    ``docs/V4_EG_RENOVATION_AUDIT.md`` Section 7.

    For ``n_regimes < 2`` the rescale is undefined, so ``base_eta`` is returned
    unchanged.
    """
    if n_regimes < 2:
        return base_eta
    return base_eta * (n_regimes ** 2 - n_regimes + 1) / (n_regimes - 1)


def _apply_v3_additive(
    affinity: Dict[str, float],
    winning_regime: str,
    regimes: List[str],
    eta: float,
) -> Dict[str, float]:
    """V3 legacy: additive update + clamp + post-hoc renormalization.

    Reproduces verbatim the update used in v1.0-v3.x. Kept for direct
    comparison runs only; do not use as the canonical update.
    """
    R = len(regimes)
    out = dict(affinity)
    for r in regimes:
        if r == winning_regime:
            out[r] = min(1.0, out.get(r, 1.0 / R) + eta)
        else:
            out[r] = max(0.01, out.get(r, 1.0 / R) - eta / (R - 1))
    total = sum(out.values())
    return {r: v / total for r, v in out.items()}


def _apply_eg(
    affinity: Dict[str, float],
    winning_regime: str,
    regimes: List[str],
    eta: float,
) -> Dict[str, float]:
    """V4 canonical: exponentiated-gradient / Hedge update.

    Multiplies the winning regime's entry by exp(eta); all entries then
    divided by the partition sum. Preserves the simplex by construction
    and the interior strictly; no clamp required.
    """
    out = dict(affinity)
    R = len(regimes)
    out[winning_regime] = out.get(winning_regime, 1.0 / R) * math.exp(eta)
    for r in regimes:
        out.setdefault(r, 1.0 / R)
    Z = sum(out.values())
    return {r: v / Z for r, v in out.items()}


def apply_affinity_update(
    affinity: Dict[str, float],
    winning_regime: str,
    regimes: List[str],
    eta: float = 0.1,
    rule: str = "eg",
) -> Dict[str, float]:
    """Apply the requested affinity update rule.

    Parameters
    ----------
    affinity : Dict[str, float]
        Current affinity distribution. Must sum to 1 (within float epsilon).
    winning_regime : str
        Regime that won this round.
    regimes : List[str]
        Canonical regime ordering.
    eta : float, default 0.1
        Learning rate. Note that V3 and V4 have different effective step
        sizes at a given ``eta``. The ``"v3_additive"`` rule implemented
        here is the *flat-additive* experiment-script variant
        (``alpha_winner += eta``), whose per-step winner gain at uniform
        start is ``eta`` while V4's is ``eta * (R - 1) / R^2``; the ratio
        is therefore ``R^2 / (R - 1)`` (~5.33 at R=4). This is distinct
        from the ``(1 - alpha)`` *paper-V3* variant in
        ``src/agents/niche_population.py::_update_niche_affinity_v3``,
        whose ratio is ``(R^2 - R + 1) / (R - 1)`` (~4.33 at R=4).
        See ``docs/V4_EG_RENOVATION_AUDIT.md`` Section 7.
    rule : {"eg", "v3_additive"}, default "eg"
        Which update rule to apply.

    Returns
    -------
    Dict[str, float]
        New affinity distribution (sums to 1).
    """
    if rule == "eg":
        return _apply_eg(affinity, winning_regime, regimes, eta)
    if rule == "v3_additive":
        return _apply_v3_additive(affinity, winning_regime, regimes, eta)
    raise ValueError(
        f"Unknown update rule {rule!r}; expected 'eg' or 'v3_additive'."
    )
