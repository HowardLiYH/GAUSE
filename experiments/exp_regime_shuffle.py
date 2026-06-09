#!/usr/bin/env python3
"""
Regime Shuffle Test - Negative Control (redesigned).

WHY THE OLD VERSION WAS WRONG
-----------------------------
The previous version shuffled the regime labels over time and then checked
whether the Specialization Index (SI) dropped. It never did -- and that is a
property of the metric, not evidence of meaningful structure. SI measures only
how *concentrated* each learner's niche-affinity distribution is. Under the
exponentiated-gradient (EG) dynamics, every learner concentrates its affinity
onto whichever regime label it keeps winning, regardless of whether that label
carries any real information. Shuffling the labels merely randomizes *which*
label a learner concentrates on; it does not reduce the *concentration*. So
SI is structurally invariant to shuffling and can never serve as a negative
control. (Confirmed empirically: shuffled SI ~= original SI in every domain.)

WHAT A REAL NEGATIVE CONTROL MUST TEST
--------------------------------------
The substantive claim is that the detected regimes are *meaningful*: knowing
the regime lets a learner pick a method that genuinely predicts better. That is
a statement about **task performance**, not about affinity concentration.

So this experiment measures the *performance benefit of regime-conditioning*:

  - ORIGINAL labels: regime r is consistently associated with data conditions
    where some method predicts best, so per-regime method selection lowers
    prediction error.
  - SHUFFLED labels: the regime label at each step is permuted, so per-regime
    method beliefs average over unrelated time steps and carry no information;
    regime-conditioned method selection should be no better than regime-blind.

Hypothesis (one-sided): mean prediction error under ORIGINAL regimes is LOWER
than under SHUFFLED regimes. If the gap is significant, the regimes are
informative for method selection. If it is ~0, the "specialization" -- however
high its SI -- does not reflect exploitable structure.

We also report SI for both conditions to make the structure-invariance of SI
explicit, so the metric is never again mistaken for a structure test.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from lambda_zero experiment
from experiments.exp_lambda_zero_real import (
    PredictionPopulation,
    get_domain_predictor,
    load_domain_data,
    bootstrap_ci,
    cohens_d,
)


def run_condition(domain: str, shuffle: bool, n_trials: int = 30,
                  n_iterations: int = 400, n_agents: int = 8,
                  seed: int = 42) -> Dict:
    """Run the population under original or shuffled regime labels.

    Returns per-trial SI and per-trial mean prediction error (over the last
    100 iterations), so the caller can test both the (structure-invariant) SI
    and the (structure-dependent) performance benefit.
    """
    values, regimes, methods = load_domain_data(domain)

    si_values: List[float] = []
    err_values: List[float] = []

    label = "shuffled" if shuffle else "original"
    for trial in tqdm(range(n_trials), desc=f"{domain} ({label})", leave=False):
        trial_seed = seed + trial * 100 + (1000 if shuffle else 0)
        rng = np.random.default_rng(trial_seed)

        if shuffle:
            # Permute the regime labels across time. The data values and the
            # predictor history are NOT shuffled -- only the label fed to the
            # population -- so any drop in performance is attributable purely to
            # the label losing its information about the underlying dynamics.
            regimes_to_use = list(rng.permutation(regimes))
        else:
            regimes_to_use = regimes

        unique_regimes = list(set(regimes))
        pop = PredictionPopulation(
            n_agents=n_agents,
            methods=methods,
            regimes=unique_regimes,
            niche_bonus_lambda=0.5,
            seed=trial_seed + 500,
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
        # Mean prediction error over the last 100 iterations (steady state).
        recent_errors = [h['mean_error'] for h in pop.history[-100:]]
        err_values.append(float(np.mean(recent_errors)))

    si_arr = np.array(si_values)
    err_arr = np.array(err_values)
    return {
        "domain": domain,
        "shuffled": shuffle,
        "n_trials": n_trials,
        "si_mean": float(si_arr.mean()),
        "si_std": float(si_arr.std()),
        "si_all": [float(x) for x in si_values],
        "error_mean": float(err_arr.mean()),
        "error_std": float(err_arr.std()),
        "error_all": [float(x) for x in err_values],
    }


def run_all_shuffle_tests(n_trials: int = 30):
    """Run the performance-based negative control on all real-data domains."""

    domains = ["energy", "weather", "finance"]

    results = {
        "experiment": "regime_shuffle_negative_control",
        "metric": "task_performance",
        "date": datetime.now().isoformat(),
        "config": {"n_trials": n_trials, "n_shuffle_seeds": 10},
        "note": (
            "Negative control measures the PERFORMANCE benefit of regime-"
            "conditioning (prediction error), not SI. SI is reported only to "
            "show it is structurally invariant to shuffling."
        ),
        "results": {},
    }

    for domain in domains:
        print(f"\n{'='*60}")
        print(f"Testing {domain.upper()}")
        print(f"{'='*60}")

        original = run_condition(domain, shuffle=False, n_trials=n_trials)

        # Pool several independent shuffles.
        shuffled_si: List[float] = []
        shuffled_err: List[float] = []
        for shuffle_seed in range(10):
            sh = run_condition(
                domain, shuffle=True, n_trials=max(1, n_trials // 3),
                seed=42 + shuffle_seed * 10000,
            )
            shuffled_si.extend(sh["si_all"])
            shuffled_err.extend(sh["error_all"])

        orig_err = np.array(original["error_all"])
        shuf_err = np.array(shuffled_err)
        shuf_si = np.array(shuffled_si)

        # One-sided test: original error should be LOWER than shuffled error.
        t_stat, p_val = stats.ttest_ind(orig_err, shuf_err, alternative='less')
        # Effect size on the error gap (shuffled - original): positive = real
        # regimes help.
        perf_effect = cohens_d(shuf_err, orig_err)
        # Relative performance benefit from real regimes.
        denom = shuf_err.mean() if shuf_err.mean() != 0 else 1e-12
        perf_gain_pct = float((shuf_err.mean() - orig_err.mean()) / abs(denom) * 100)

        results["results"][domain] = {
            "original": {
                "si_mean": original["si_mean"], "si_std": original["si_std"],
                "error_mean": original["error_mean"], "error_std": original["error_std"],
            },
            "shuffled": {
                "si_mean": float(shuf_si.mean()), "si_std": float(shuf_si.std()),
                "error_mean": float(shuf_err.mean()), "error_std": float(shuf_err.std()),
                "n_samples": len(shuffled_err),
            },
            "performance_test": {
                "t_statistic": float(t_stat),
                "p_value_one_sided": float(p_val),
                "cohens_d": float(perf_effect),
                "significant_001": bool(p_val < 0.001),
                "perf_gain_pct": perf_gain_pct,
            },
            "si_invariance": {
                "si_drop_pct": float(
                    (original["si_mean"] - shuf_si.mean()) / original["si_mean"] * 100
                ),
                "note": "Near-zero by construction; SI does not test structure.",
            },
        }

        c = results["results"][domain]
        print(f"  SI:    original {original['si_mean']:.3f} | shuffled {shuf_si.mean():.3f}  "
              f"(drop {c['si_invariance']['si_drop_pct']:+.1f}% -- expected ~0)")
        print(f"  Error: original {original['error_mean']:.4f} | shuffled {shuf_err.mean():.4f}")
        print(f"  Regime-conditioning performance benefit: {perf_gain_pct:+.1f}%")
        print(f"  t={t_stat:.2f}, p(one-sided)={p_val:.2e}, d={perf_effect:.2f} "
              f"{'[SIGNIFICANT]' if p_val < 0.001 else '[n.s.]'}")

    # Summary
    print("\n" + "="*90)
    print("SUMMARY: Regime Shuffle Negative Control (performance-based)")
    print("="*90)
    print(f"{'Domain':<12} {'SI orig':>9} {'SI shuf':>9} {'Err orig':>12} {'Err shuf':>12} "
          f"{'Benefit':>9} {'p':>10} {'Sig?':>6}")
    print("-"*90)

    all_significant = True
    for domain in domains:
        r = results["results"][domain]
        sig = r["performance_test"]["significant_001"]
        all_significant = all_significant and sig
        print(f"{domain:<12} {r['original']['si_mean']:>9.3f} {r['shuffled']['si_mean']:>9.3f} "
              f"{r['original']['error_mean']:>12.4f} {r['shuffled']['error_mean']:>12.4f} "
              f"{r['performance_test']['perf_gain_pct']:>8.1f}% "
              f"{r['performance_test']['p_value_one_sided']:>10.2e} {'OK' if sig else 'x':>6}")
    print("="*90)

    if all_significant:
        print("\nAll domains: regime-conditioning significantly lowers prediction error"
              "\nunder real labels and that benefit vanishes under shuffling.")
        print("=> The detected regimes carry exploitable structure (control PASSES).")
    else:
        print("\nNot all domains show a significant performance benefit from real regimes.")
        print("=> Where the benefit is absent, high SI does NOT reflect exploitable"
              "\n   structure -- it is step-size-driven concentration. Report honestly.")

    print("\nNote: SI is ~unchanged by shuffling in all domains -- it measures"
          "\nconcentration, not structure. That is why this control uses performance.")

    output_dir = Path(__file__).parent.parent / "results" / "regime_shuffle"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return results


if __name__ == "__main__":
    print("="*80)
    print("Regime Shuffle Test - Negative Control (performance-based)")
    print("Testing whether detected regimes are informative for method selection")
    print("="*80)

    results = run_all_shuffle_tests(n_trials=30)
