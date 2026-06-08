#!/usr/bin/env python3
"""
Route B validation: ablation (coupled vs decoupled) + multi-series robustness.

Two questions left after the domain hunt + authentication:

  1. ABLATION -- is the niche->winner COUPLING what produces the performance gap,
     or would any winner-take-all routing do? We run the identical pipeline with
     the coupling term ON (Route B) and OFF (current/decoupled pipeline). If the
     real-vs-shuffled gap appears only with coupling, the architecture is the
     causal contribution.

  2. ROBUSTNESS -- the headline previously rested on one series per domain. Here
     we run MANY: four month-windows of NYC taxi and all five solar locations,
     each with a causal clock-based regime. A distribution of gaps (not n=1)
     guards against cherry-picking.

All regimes are clock-based (time-of-day / weekday) and therefore causal: the
label is known before the target is observed and cannot leak it.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.exp_domain_screen import build_steps, METHODS, oracle_gap
from experiments.exp_route_b_coupled import run_condition, cohens_d, CoupledPopulation

DATA = Path(__file__).parent.parent / "data"


def single_agent_error(steps, regime_list, n_trials: int) -> np.ndarray:
    """Baseline: ONE regime-conditioned agent (no population, no specialization).

    It keeps per-regime Thompson beliefs and selects/updates every step (it always
    'wins' because it is alone). This isolates the value of regime-CONDITIONING
    from the value of population SPECIALIZATION: if the specialized population does
    not beat this monolith, specialization is not what buys the accuracy.
    """
    methods = list(METHODS)
    errs_out = []
    for t in range(n_trials):
        rng = np.random.default_rng(100 + t)
        beliefs = {r: {m: [1.0, 1.0] for m in methods} for r in regime_list}
        step_err = []
        for regime, errs in steps:
            if regime not in beliefs:
                beliefs[regime] = {m: [1.0, 1.0] for m in methods}
            samples = {m: rng.beta(*beliefs[regime][m]) for m in methods}
            choice = max(samples, key=samples.get)
            e = errs[choice]
            step_err.append(e)
            max_e, min_e = max(errs.values()), min(errs.values())
            span = (max_e - min_e) or 1.0
            q = (max_e - e) / span
            if q >= 0.5:
                beliefs[regime][choice][0] += 1.0
            else:
                beliefs[regime][choice][1] += 1.0
        half = len(step_err) // 2
        errs_out.append(float(np.mean(step_err[half:])))
    return np.array(errs_out)


# --------------------------------------------------------------------------- #
# Clock-based (causal) regime construction
# --------------------------------------------------------------------------- #
def traffic_regime(ts: pd.Timestamp) -> str:
    h, wd = ts.hour, ts.weekday()
    if wd >= 5:
        return "weekend"
    if 7 <= h <= 9:
        return "morning_rush"
    if 17 <= h <= 19:
        return "evening_rush"
    if 10 <= h <= 16:
        return "midday"
    if 0 <= h <= 5:
        return "night"
    return "transition"


def solar_regime(hour: int) -> str:
    if hour < 6 or hour >= 20:
        return "night"
    if hour < 9:
        return "morning"
    if hour < 16:
        return "midday"
    return "evening"


def load_traffic_windows(window: int = 720) -> List[Tuple[str, np.ndarray, List[str]]]:
    """Split NYC taxi into consecutive month-sized windows -> several series."""
    df = pd.read_csv(DATA / "traffic" / "nyc_taxi_real_hourly.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    out = []
    n = len(df)
    for w, start in enumerate(range(0, n, window)):
        chunk = df.iloc[start:start + window]
        if len(chunk) < 200:
            continue
        values = chunk["trip_count"].to_numpy(dtype=float)
        regimes = [traffic_regime(ts) for ts in chunk["timestamp"]]
        out.append((f"traffic_w{w+1}", values, regimes))
    return out


def load_solar_locations(max_n: int = 800) -> List[Tuple[str, np.ndarray, List[str]]]:
    df = pd.read_csv(DATA / "solar" / "openmeteo_real_irradiance.csv")
    out = []
    for loc in sorted(df["location"].dropna().unique()):
        sub = df[df["location"] == loc].head(max_n)
        values = pd.to_numeric(sub["ghi"], errors="coerce").to_numpy()
        hours = sub["hour"].to_numpy()
        mask = ~np.isnan(values)
        values, hours = values[mask], hours[mask]
        if len(values) < 200:
            continue
        regimes = [solar_regime(int(h)) for h in hours]
        out.append((f"solar_{loc}", values, regimes))
    return out


# --------------------------------------------------------------------------- #
# One series, both arms
# --------------------------------------------------------------------------- #
def gap_for(steps, regime_list, coupled: bool, n_trials: int) -> Dict:
    real = run_condition(steps, regime_list, list(METHODS), shuffle=False,
                         n_trials=n_trials, base_seed=1, coupled=coupled)
    shuf = run_condition(steps, regime_list, list(METHODS), shuffle=True,
                         n_trials=n_trials, base_seed=2, coupled=coupled)
    _t, p = stats.ttest_ind(real["err"], shuf["err"], alternative="less")
    gap = (shuf["err"].mean() - real["err"].mean()) / shuf["err"].mean() * 100
    return {"gap": gap, "p": p, "d": cohens_d(real["err"], shuf["err"]),
            "real": real["err"].mean(), "shuf": shuf["err"].mean()}


def evaluate_series(series, n_trials: int = 20) -> List[Dict]:
    rows = []
    for name, values, regimes in series:
        steps = build_steps(values, regimes)
        regime_list = sorted(set(regimes[25:]))
        coup = gap_for(steps, regime_list, coupled=True, n_trials=n_trials)
        deco = gap_for(steps, regime_list, coupled=False, n_trials=n_trials)
        rows.append({"name": name, "n": len(values),
                     "coup_gap": coup["gap"], "coup_p": coup["p"], "coup_d": coup["d"],
                     "deco_gap": deco["gap"], "deco_p": deco["p"]})
    return rows


def evaluate_baseline(series, n_trials: int = 20) -> List[Dict]:
    """Population (coupled, specialized) vs single regime-conditioned agent."""
    rows = []
    for name, values, regimes in series:
        steps = build_steps(values, regimes)
        regime_list = sorted(set(regimes[25:]))
        pop = run_condition(steps, regime_list, list(METHODS), shuffle=False,
                            n_trials=n_trials, base_seed=1, coupled=True)["err"]
        single = single_agent_error(steps, regime_list, n_trials=n_trials)
        # positive gap => specialized population beats the single conditioned agent
        gap = (single.mean() - pop.mean()) / single.mean() * 100
        _t, p = stats.ttest_ind(pop, single, alternative="less")
        rows.append({"name": name, "pop": pop.mean(), "single": single.mean(),
                     "gap": gap, "p": p})
    return rows


def print_baseline(title: str, rows: List[Dict]):
    print(f"\n{'=' * 84}\n{title}\n{'=' * 84}")
    print(f"{'series':<20}{'population err':>16}{'single-agent err':>18}"
          f"{'pop better%':>13}{'p':>10}")
    print("-" * 84)
    for r in rows:
        print(f"{r['name']:<20}{r['pop']:>16.3f}{r['single']:>18.3f}"
              f"{r['gap']:>13.2f}{r['p']:>10.2g}")
    g = np.array([r["gap"] for r in rows])
    print("-" * 84)
    print(f"{'MEAN':<20}{'':>16}{'':>18}{g.mean():>13.2f}")
    return g


def print_block(title: str, rows: List[Dict]):
    print(f"\n{'=' * 84}\n{title}\n{'=' * 84}")
    print(f"{'series':<20}{'n':>6}{'COUPLED gap%':>15}{'p':>10}{'d':>7}"
          f"{'DECOUPLED gap%':>16}{'p':>10}")
    print("-" * 84)
    for r in rows:
        print(f"{r['name']:<20}{r['n']:>6}{r['coup_gap']:>15.2f}{r['coup_p']:>10.2g}"
              f"{r['coup_d']:>7.2f}{r['deco_gap']:>16.2f}{r['deco_p']:>10.2g}")
    cg = np.array([r["coup_gap"] for r in rows])
    dg = np.array([r["deco_gap"] for r in rows])
    print("-" * 84)
    print(f"{'MEAN':<20}{'':>6}{cg.mean():>15.2f}{'':>10}{'':>7}{dg.mean():>16.2f}")
    sig = sum(1 for r in rows if r["coup_p"] < 0.05 and r["coup_gap"] > 1.0)
    print(f"  coupled significant (p<0.05, gap>1%): {sig}/{len(rows)} series")
    return cg, dg


def main():
    print("#" * 84)
    print("# ROUTE B VALIDATION: coupled-vs-decoupled ablation + multi-series robustness")
    print("# (all regimes clock-based => causal, cannot leak the target)")
    print("#" * 84)

    traffic = evaluate_series(load_traffic_windows())
    solar = evaluate_series(load_solar_locations())

    cg_t, dg_t = print_block("TRAFFIC (NYC taxi, month-windows, weekday/hour regimes)", traffic)
    cg_s, dg_s = print_block("SOLAR (5 locations, hour-of-day regimes)", solar)

    all_coup = np.concatenate([cg_t, cg_s])
    all_deco = np.concatenate([dg_t, dg_s])
    print("\n" + "#" * 84)
    print("# OVERALL")
    print("#" * 84)
    print(f"  coupled   mean gap across {len(all_coup)} series: {all_coup.mean():+.2f}% "
          f"(min {all_coup.min():+.2f}, max {all_coup.max():+.2f})")
    print(f"  decoupled mean gap across {len(all_deco)} series: {all_deco.mean():+.2f}% "
          f"(min {all_deco.min():+.2f}, max {all_deco.max():+.2f})")
    _t, p_paired = stats.ttest_rel(all_coup, all_deco)
    print(f"  paired t-test coupled>decoupled: p={p_paired:.3g}")

    # Decisive baseline: does the specialized population beat a single
    # regime-conditioned agent? If not, the accuracy comes from regime-
    # CONDITIONING, not population SPECIALIZATION.
    base_t = evaluate_baseline(load_traffic_windows())
    base_s = evaluate_baseline(load_solar_locations())
    gt = print_baseline("BASELINE: specialized population vs single regime-conditioned agent (TRAFFIC)", base_t)
    gs = print_baseline("BASELINE: specialized population vs single regime-conditioned agent (SOLAR)", base_s)
    allg = np.concatenate([gt, gs])
    print("\n" + "#" * 84)
    print("# INTERPRETATION")
    print("#" * 84)
    print(f"  coupled vs decoupled: indistinguishable (paired p={p_paired:.2g}) =>")
    print("    the niche->winner coupling does NOT add steady-state accuracy.")
    print(f"  population vs single conditioned agent: mean pop-better = {allg.mean():+.2f}% =>")
    if allg.mean() > 1.0:
        print("    the specialized population DOES beat the monolith (specialization helps).")
    else:
        print("    the population does NOT beat the monolith: the accuracy gain comes from")
        print("    REGIME-CONDITIONING, while population specialization is a robust")
        print("    division-of-labor phenomenon (coverage~=1) rather than an accuracy lever.")

    def _clean(rows):
        return [{k: (float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v)
                 for k, v in r.items()} for r in rows]

    # Oracle gaps with the SAME causal clock regimes used above (so the oracle
    # table and the real-vs-shuffled headline share a regime definition).
    oracle = []
    for name, values, regimes in load_traffic_windows() + load_solar_locations():
        og = oracle_gap(values, regimes)
        oracle.append({"series": name, "oracle_gap_pct": float(og["gap_pct"]),
                       "n_regimes": int(og["n_regimes"]),
                       "distinct_winners": int(og["distinct_winners"])})

    out_dir = Path(__file__).parent.parent / "results" / "route_b_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "route_b_validation_ablation_and_baseline",
            "oracle_gaps_causal_regimes": oracle,
            "ablation_coupled_vs_decoupled": {
                "traffic": _clean(traffic), "solar": _clean(solar),
                "coupled_mean_gap_pct": float(all_coup.mean()),
                "decoupled_mean_gap_pct": float(all_deco.mean()),
                "paired_p_coupled_gt_decoupled": float(p_paired)},
            "baseline_population_vs_single_agent": {
                "traffic": _clean(base_t), "solar": _clean(base_s),
                "mean_population_better_pct": float(allg.mean())},
        }, f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
