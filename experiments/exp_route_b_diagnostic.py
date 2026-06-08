#!/usr/bin/env python3
"""
Route B gate: is there ANY exploitable regime structure?

Before redesigning the algorithm to couple niche specialization to performance,
we must know whether regime-specific method specialization could help *even in
principle*. This is an oracle analysis that ignores the learning algorithm
entirely and asks a property of the DATA + PREDICTORS:

  - GENERALIST oracle: pick the single best method overall, use it everywhere.
  - SPECIALIST oracle: for each regime, pick that regime's best method, and
    switch methods as the true regime changes.

If SPECIALIST << GENERALIST, then different regimes genuinely favor different
methods -> there is structure for a regime-aware algorithm to exploit -> Route B
is worth building. If SPECIALIST ~= GENERALIST, no algorithm (no matter how it
couples niche to behavior) can extract a regime benefit, because the regimes do
not change which method is best -> go straight to Route A (division of labor).

This is the cheapest possible test of the strong thesis's *upper bound*.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.exp_lambda_zero_real import load_domain_data, get_domain_predictor

DOMAINS = ["energy", "weather", "finance"]


def analyze_domain(domain: str) -> Dict:
    values, regimes, methods = load_domain_data(domain)
    n = len(values)

    # error[idx][method] = |prediction - truth| ; regime_at[idx]
    per_method_errors = defaultdict(list)          # method -> [errors over all steps]
    per_regime_method_errors = defaultdict(lambda: defaultdict(list))  # regime -> method -> [errors]
    step_errors = []                               # list of (regime, {method: error})

    for idx in range(20, n):
        history = values[:idx]
        true_val = values[idx]
        regime = regimes[idx]
        errs = {}
        for m in methods:
            pred = get_domain_predictor(domain, m, history, idx)
            e = abs(pred - true_val)
            errs[m] = e
            per_method_errors[m].append(e)
            per_regime_method_errors[regime][m].append(e)
        step_errors.append((regime, errs))

    # Generalist oracle: single globally-best method.
    method_mean = {m: float(np.mean(per_method_errors[m])) for m in methods}
    best_global = min(method_mean, key=method_mean.get)
    generalist_err = float(np.mean([errs[best_global] for _, errs in step_errors]))

    # Specialist oracle: best method per regime.
    best_per_regime = {}
    per_regime_method_mean = {}
    for regime, mdict in per_regime_method_errors.items():
        regime_method_mean = {m: float(np.mean(mdict[m])) for m in mdict}
        per_regime_method_mean[regime] = regime_method_mean
        best_per_regime[regime] = min(regime_method_mean, key=regime_method_mean.get)
    specialist_err = float(np.mean([errs[best_per_regime[regime]] for regime, errs in step_errors]))

    improvement_pct = (generalist_err - specialist_err) / generalist_err * 100 if generalist_err else 0.0

    # Do regimes actually disagree about the best method?
    distinct_best = set(best_per_regime.values())
    regimes_disagree = len(distinct_best) > 1

    return {
        "domain": domain,
        "methods": methods,
        "regime_counts": {r: len(per_regime_method_errors[r][methods[0]]) for r in per_regime_method_errors},
        "global_best_method": best_global,
        "best_per_regime": best_per_regime,
        "generalist_err": generalist_err,
        "specialist_err": specialist_err,
        "improvement_pct": improvement_pct,
        "regimes_disagree_on_best_method": regimes_disagree,
        "method_mean_error": method_mean,
        "per_regime_method_mean": per_regime_method_mean,
    }


def main():
    print("=" * 88)
    print("ROUTE B GATE: is there exploitable regime structure? (oracle analysis)")
    print("=" * 88)

    any_structure = False
    saved = []
    for domain in DOMAINS:
        r = analyze_domain(domain)
        saved.append({"domain": r["domain"], "global_best_method": r["global_best_method"],
                      "best_per_regime": r["best_per_regime"],
                      "generalist_err": r["generalist_err"], "specialist_err": r["specialist_err"],
                      "improvement_pct": r["improvement_pct"],
                      "regimes_disagree_on_best_method": r["regimes_disagree_on_best_method"]})
        print(f"\nDOMAIN: {domain.upper()}")
        print(f"  methods: {r['methods']}")
        print(f"  per-regime method error (lower=better; * = regime winner):")
        header = "      " + f"{'regime':<16}" + "".join(f"{m:>16}" for m in r["methods"])
        print(header)
        for regime in r["best_per_regime"]:
            cnt = r["regime_counts"].get(regime, 0)
            cells = ""
            for m in r["methods"]:
                val = r["per_regime_method_mean"][regime][m]
                mark = "*" if m == r["best_per_regime"][regime] else " "
                cells += f"{mark}{val:>14.3f} "
            print(f"      {regime:<16}{cells}(n={cnt})")
        print(f"  global best method:      {r['global_best_method']}")
        print(f"  GENERALIST oracle error: {r['generalist_err']:.4f}")
        print(f"  SPECIALIST oracle error: {r['specialist_err']:.4f}")
        print(f"  specialist improvement:  {r['improvement_pct']:+.2f}%")
        print(f"  regimes disagree on best method: {r['regimes_disagree_on_best_method']}")
        if r["regimes_disagree_on_best_method"] and r["improvement_pct"] > 1.0:
            any_structure = True
            print("  => exploitable structure PRESENT in this domain.")
        else:
            print("  => little/no exploitable regime structure here.")

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if any_structure:
        print("At least one domain has exploitable regime structure (specialist oracle")
        print("beats generalist). Route B (couple niche -> behavior) is worth building:")
        print("a regime-aware algorithm could in principle capture that gap.")
    else:
        print("NO domain shows a meaningful specialist-vs-generalist gap. The regimes do")
        print("not change which method is best, so no coupling can produce a performance")
        print("benefit. Route B cannot rescue the strong thesis here -> pivot to Route A")
        print("(coordination-free division of labor), and/or find domains/predictors with")
        print("genuine regime-dependent method optimality.")

    out_dir = Path(__file__).parent.parent / "results" / "route_b_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"experiment": "route_b_oracle_gate",
                   "note": "specialist-vs-generalist oracle gap (energy/weather/finance)",
                   "any_exploitable_structure": any_structure, "domains": saved}, f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
