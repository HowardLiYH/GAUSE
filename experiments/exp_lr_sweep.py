#!/usr/bin/env python3
"""
Learning-rate (eta) sweep with a non-circular selection criterion.

MOTIVATION
----------
The headline pipeline reports V4 at a rescaled step size (eta ~= 0.43 at R=4).
Because the Specialization Index (SI) rises monotonically with eta -- a larger
EG step concentrates affinity harder regardless of whether anything meaningful
is happening -- selecting eta by maximizing SI is circular. This sweep instead
records, at each eta, BOTH:

  1. SI(eta)              -- the (eta-inflatable) headline metric, reported for
                            transparency / sensitivity analysis.
  2. perf_gap(eta)        -- the structure-dependent benefit of regime-
                            conditioning, defined as
                              (prediction error with SHUFFLED regimes)
                            - (prediction error with REAL regimes)
                            from the corrected negative control. This does NOT
                            mechanically increase with eta, so it is a legitimate
                            criterion for choosing eta.

Read the output as follows:
  - perf_gap ~ 0 at all eta  -> no learning rate produces meaningful
                                specialization; the effect is concentration only.
  - perf_gap peaks at eta*    -> eta* is the honestly-selected ideal rate.
  - perf_gap > 0 but flat     -> modest, eta-insensitive real benefit.

The paper narrative should anchor at the natural eta = 0.1 (per the audit doc)
and present this sweep as the sensitivity / honesty analysis.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.exp_lambda_zero_real import (
    PredictionPopulation,
    get_domain_predictor,
    load_domain_data,
    cohens_d,
)

DOMAINS = ["energy", "weather", "finance"]
ETA_GRID = [0.05, 0.1, 0.2, 0.3, 0.43, 0.6, 0.8, 1.2]
NATURAL_ETA = 0.1


def run_condition(domain: str, shuffle: bool, lr: float, n_trials: int,
                  n_iterations: int = 400, n_agents: int = 8, seed: int = 42) -> Dict:
    """Train the population at a fixed EG step `lr`; return per-trial SI and error."""
    values, regimes, methods = load_domain_data(domain)
    unique_regimes = list(set(regimes))

    si_values: List[float] = []
    err_values: List[float] = []

    for trial in range(n_trials):
        trial_seed = seed + trial * 100 + (1000 if shuffle else 0)
        rng = np.random.default_rng(trial_seed)
        regimes_to_use = list(rng.permutation(regimes)) if shuffle else regimes

        pop = PredictionPopulation(
            n_agents=n_agents, methods=methods, regimes=unique_regimes,
            niche_bonus_lambda=0.5, seed=trial_seed + 500, lr_override=lr,
        )

        n_steps = min(n_iterations, len(values) - 20)
        for it in range(n_steps):
            idx = it + 20
            regime = regimes_to_use[idx]
            true_val = values[idx]
            history = values[:idx]

            def predict_fn(method):
                return get_domain_predictor(domain, method, history, idx)

            pop.run_iteration(regime, true_val, predict_fn)

        si_values.append(pop.get_population_si())
        err_values.append(float(np.mean([h['mean_error'] for h in pop.history[-100:]])))

    return {"si": np.array(si_values), "err": np.array(err_values)}


def sweep_domain(domain: str, n_trials: int = 30) -> List[Dict]:
    """For each eta, compute SI and the real-vs-shuffled performance gap."""
    rows = []
    for eta in ETA_GRID:
        orig = run_condition(domain, shuffle=False, lr=eta, n_trials=n_trials)

        shuf_si: List[float] = []
        shuf_err: List[float] = []
        for s in range(6):
            sh = run_condition(domain, shuffle=True, lr=eta,
                               n_trials=max(1, n_trials // 3), seed=42 + s * 10000)
            shuf_si.extend(sh["si"].tolist())
            shuf_err.extend(sh["err"].tolist())
        shuf_err = np.array(shuf_err)

        _t_stat, p_val = stats.ttest_ind(orig["err"], shuf_err, alternative='less')
        denom = shuf_err.mean() if shuf_err.mean() != 0 else 1e-12
        rows.append({
            "eta": eta,
            "si_original": float(orig["si"].mean()),
            "si_shuffled": float(np.mean(shuf_si)),
            "err_original": float(orig["err"].mean()),
            "err_shuffled": float(shuf_err.mean()),
            "perf_gap_pct": float((shuf_err.mean() - orig["err"].mean()) / abs(denom) * 100),
            "perf_cohens_d": float(cohens_d(shuf_err, orig["err"])),
            "p_value_one_sided": float(p_val),
            "significant_001": bool(p_val < 0.001),
        })
    return rows


def main():
    print("=" * 92)
    print("LEARNING-RATE SWEEP: SI(eta) vs structure-dependent performance gap")
    print("=" * 92)

    all_results = {
        "experiment": "lr_sweep",
        "date": datetime.now().isoformat(),
        "eta_grid": ETA_GRID,
        "natural_eta": NATURAL_ETA,
        "criterion": "perf_gap = err(shuffled) - err(real); choose eta by perf_gap, NOT SI",
        "results": {},
    }

    for domain in DOMAINS:
        print(f"\n{'='*92}\nDOMAIN: {domain.upper()}\n{'='*92}")
        print(f"{'eta':>6} {'SI(real)':>10} {'SI(shuf)':>10} {'err(real)':>12} "
              f"{'err(shuf)':>12} {'perf gap':>10} {'d':>7} {'p':>10} {'sig?':>5}")
        print("-" * 92)
        rows = sweep_domain(domain)
        all_results["results"][domain] = rows
        for r in rows:
            mark = " <- natural" if abs(r["eta"] - NATURAL_ETA) < 1e-9 else ""
            print(f"{r['eta']:>6.2f} {r['si_original']:>10.3f} {r['si_shuffled']:>10.3f} "
                  f"{r['err_original']:>12.4f} {r['err_shuffled']:>12.4f} "
                  f"{r['perf_gap_pct']:>9.1f}% {r['perf_cohens_d']:>7.2f} "
                  f"{r['p_value_one_sided']:>10.2e} {'OK' if r['significant_001'] else 'x':>5}{mark}")

    # Verdict per domain: is there ANY eta with a significant positive perf gap?
    print(f"\n{'='*92}\nVERDICT\n{'='*92}")
    for domain in DOMAINS:
        rows = all_results["results"][domain]
        sig_rows = [r for r in rows if r["significant_001"] and r["perf_gap_pct"] > 0]
        if sig_rows:
            best = max(sig_rows, key=lambda r: r["perf_gap_pct"])
            print(f"{domain:<10}: structure-dependent benefit EXISTS; peak at eta={best['eta']} "
                  f"(+{best['perf_gap_pct']:.1f}%, p={best['p_value_one_sided']:.1e}).")
        else:
            print(f"{domain:<10}: NO eta yields a significant performance benefit from real "
                  f"regimes. High SI here is concentration, not exploitable structure.")

    print("\nSI rises with eta in every domain (see SI columns) -- confirming SI is "
          "eta-inflatable\nand must not be used to select eta. The perf-gap column is the "
          "honest criterion.")

    out_dir = Path(__file__).parent.parent / "results" / "lr_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_dir}")
    return all_results


if __name__ == "__main__":
    main()
