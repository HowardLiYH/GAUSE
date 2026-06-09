#!/usr/bin/env python3
"""
Domain screener: find real datasets with genuine, exploitable regime structure.

The oracle gate (exp_route_b_diagnostic.py) showed that energy/weather have a
globally-dominant method (specialist-vs-generalist ceiling = 0%) and finance has
only ~1.9%. To support the STRONG thesis we need real domains where different
regimes genuinely favor different methods -- i.e. a large specialist-oracle gap.

This screener sweeps a catalogue of real time series, applies a shared set of
generic deterministic forecasting methods, and labels regimes two ways:

  * NATIVE   -- the dataset's own ``regime`` column, if present.
  * data-driven schemes (trend / volatility / level tertiles).

For every (dataset x regime-scheme) it computes the oracle gap
``(generalist_err - specialist_err) / generalist_err`` and ranks them. Large
gaps (rule of thumb >5%) are promising materials for the strong claim; the
coupled algorithm (exp_route_b_coupled.py) is then expected to capture them and
to collapse under label shuffling.

All methods are deterministic so the oracle reflects the data, not RNG noise.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Callable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA = Path(__file__).parent.parent / "data"


# --------------------------------------------------------------------------- #
# Generic deterministic forecasting methods: history -> next-value prediction
# --------------------------------------------------------------------------- #
def m_persistence(h: np.ndarray) -> float:
    return h[-1]


def m_momentum(h: np.ndarray) -> float:
    if len(h) >= 5:
        return h[-1] + 0.5 * (h[-1] - h[-5])
    return h[-1]


def m_mean_revert(h: np.ndarray) -> float:
    if len(h) >= 20:
        return h[-1] + 0.3 * (np.mean(h[-20:]) - h[-1])
    return h[-1]


def m_ma_short(h: np.ndarray) -> float:
    return float(np.mean(h[-3:]))


def m_ma_long(h: np.ndarray) -> float:
    return float(np.mean(h[-20:])) if len(h) >= 20 else float(np.mean(h))


def m_drift(h: np.ndarray) -> float:
    if len(h) >= 10:
        return h[-1] + float(np.mean(np.diff(h[-10:])))
    return h[-1]


METHODS: Dict[str, Callable[[np.ndarray], float]] = {
    "persistence": m_persistence,
    "momentum": m_momentum,
    "mean_revert": m_mean_revert,
    "ma_short": m_ma_short,
    "ma_long": m_ma_long,
    "drift": m_drift,
}


# --------------------------------------------------------------------------- #
# Data-driven regime labelers: values -> list of labels (one per index).
#
# CAUSAL by construction: the label for step i is a function of values[:i] only
# (information available BEFORE observing values[i], the prediction target).
# Thresholds use EXPANDING (past-only) percentiles, never the full sample. This
# is required for the regime to be a legitimate routing signal -- a label that
# peeks at values[i] would leak the target and inflate the real-vs-shuffled gap.
# --------------------------------------------------------------------------- #
def regimes_trend(values: np.ndarray, k: int = 5) -> List[str]:
    labels = []
    for i in range(len(values)):
        if i < k + 1:
            labels.append("flat")
            continue
        # last observed return, ending at i-1 (no peek at values[i])
        ret = (values[i - 1] - values[i - 1 - k]) / (abs(values[i - 1 - k]) + 1e-9)
        if ret > 0.01:
            labels.append("up")
        elif ret < -0.01:
            labels.append("down")
        else:
            labels.append("flat")
    return labels


def regimes_volatility(values: np.ndarray, win: int = 20) -> List[str]:
    n = len(values)
    # vol[i] = std of the window ENDING at i-1 (strictly past).
    vol = np.zeros(n)
    for i in range(2, n):
        vol[i] = np.std(values[max(0, i - win):i])
    labels = []
    for i in range(n):
        if i < win + 5:
            labels.append("mid_vol")
            continue
        lo, hi = np.percentile(vol[2:i], [33, 66])  # expanding, past-only
        labels.append("low_vol" if vol[i] <= lo else ("high_vol" if vol[i] >= hi else "mid_vol"))
    return labels


def regimes_level(values: np.ndarray) -> List[str]:
    labels = []
    for i in range(len(values)):
        if i < 30:
            labels.append("mid")
            continue
        lo, hi = np.percentile(values[:i], [33, 66])  # expanding, past-only
        last = values[i - 1]  # last observed value, not values[i]
        labels.append("low" if last <= lo else ("high" if last >= hi else "mid"))
    return labels


DATA_DRIVEN = {
    "trend": regimes_trend,
    "volatility": regimes_volatility,
    "level": regimes_level,
}


# --------------------------------------------------------------------------- #
# Oracle gap
# --------------------------------------------------------------------------- #
def oracle_gap(values: np.ndarray, regimes: List[str], warmup: int = 25) -> Dict:
    per_method_err = defaultdict(list)
    per_regime_method_err = defaultdict(lambda: defaultdict(list))
    step_errs = []
    for idx in range(warmup, len(values)):
        h = values[:idx]
        true = values[idx]
        errs = {name: abs(fn(h) - true) for name, fn in METHODS.items()}
        step_errs.append((regimes[idx], errs))
        for name, e in errs.items():
            per_method_err[name].append(e)
            per_regime_method_err[regimes[idx]][name].append(e)

    method_mean = {m: float(np.mean(per_method_err[m])) for m in METHODS}
    best_global = min(method_mean, key=method_mean.get)
    generalist = float(np.mean([errs[best_global] for _, errs in step_errs]))

    best_per_regime = {}
    for r, md in per_regime_method_err.items():
        rm = {m: float(np.mean(md[m])) for m in md}
        best_per_regime[r] = min(rm, key=rm.get)
    specialist = float(np.mean([errs[best_per_regime[r]] for r, errs in step_errs]))

    gap = (generalist - specialist) / generalist * 100 if generalist else 0.0
    return {
        "gap_pct": gap,
        "generalist_err": generalist,
        "specialist_err": specialist,
        "best_global": best_global,
        "best_per_regime": best_per_regime,
        "n_regimes": len(best_per_regime),
        "distinct_winners": len(set(best_per_regime.values())),
    }


# --------------------------------------------------------------------------- #
# Candidate datasets:
#   (label, csv, value_col, group_col, group_val, regime_col, native_causal)
# native_causal=True only when the dataset's own regime label is derived from
# information available BEFORE the target (e.g. clock-based). Traffic regimes are
# pure time-of-day/weekday (causal). Copper/solar/air-quality native regimes are
# computed from the CURRENT row's value (clear-sky index, price-vs-MA, AQI) and
# global percentiles -> they would leak the target, so we do NOT trust them and
# rely on the causal data-driven schemes instead.
# --------------------------------------------------------------------------- #
CANDIDATES = [
    ("traffic_nyc",    "traffic/nyc_taxi_real_hourly.csv", "trip_count", None, None, "regime", True),
    ("solar_denver",   "solar/openmeteo_real_irradiance.csv", "ghi", "location", "Denver_CO", "regime", False),
    ("airq_la",        "air_quality/epa_daily_aqi.csv", "pm25", "city", "Los_Angeles", "regime", False),
    ("commodity_copper","commodities/fred_real_prices.csv", "price", "commodity", "Copper", "regime", False),
    ("weather_chicago","weather/openmeteo_real_weather.csv", "temperature", "city", "Chicago", "regime", False),
    ("btc_4h",         "bybit/BTCUSDT_4H.csv", "close", None, None, None, False),
    ("eth_1h",         "bybit/ETHUSDT_1H.csv", "close", None, None, None, False),
    ("doge_4h",        "bybit/DOGEUSDT_4H.csv", "close", None, None, None, False),
    ("sol_4h",         "bybit/SOLUSDT_4H.csv", "close", None, None, None, False),
]


def load_series(csv: str, value_col: str, group_col: Optional[str],
                group_val: Optional[str], regime_col: Optional[str],
                max_n: int = 800):
    path = DATA / csv
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if group_col and group_col in df.columns and group_val is not None:
        df = df[df[group_col] == group_val]
    df = df.head(max_n)
    if value_col not in df.columns:
        return None
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy()
    mask = ~np.isnan(values)
    values = values[mask]
    native = None
    if regime_col and regime_col in df.columns:
        native = list(np.array(df[regime_col].astype(str))[mask])
    if len(values) < 100:
        return None
    return values, native


def build_steps(values: np.ndarray, regimes: List[str], warmup: int = 25):
    """Build coupled-population steps (regime, {method: error}) from a series."""
    steps = []
    for idx in range(warmup, len(values)):
        h = values[:idx]
        true = values[idx]
        errs = {name: abs(fn(h) - true) for name, fn in METHODS.items()}
        steps.append((regimes[idx], errs))
    return steps


def validate_candidates(promising: List[Dict]) -> List[Dict]:
    """Run the coupled algorithm on promising candidates: real vs shuffled."""
    from experiments.exp_route_b_coupled import evaluate

    print("\n" + "=" * 92)
    print("VALIDATION: does the COUPLED algorithm capture these real-data gaps?")
    print("(real << shuffled with p<0.05 => strong claim supported on REAL data)")
    print("=" * 92)

    validation = []
    seen = set()
    for r in promising:
        key = (r["dataset"], r["scheme"])
        if key in seen:
            continue
        seen.add(key)
        cand = next(c for c in CANDIDATES if c[0] == r["dataset"])
        _, csv, vcol, gcol, gval, rcol, _native_causal = cand
        loaded = load_series(csv, vcol, gcol, gval, rcol)
        if loaded is None:
            continue
        values, native = loaded
        labels = native if r["scheme"] == "native" else DATA_DRIVEN[r["scheme"]](values)
        steps = build_steps(values, labels)
        regime_list = sorted(set(labels[25:]))
        res = evaluate(f"{r['dataset']} / {r['scheme']} (oracle gap {r['gap_pct']:.1f}%)",
                       steps, regime_list, list(METHODS), n_trials=20)
        validation.append({"dataset": r["dataset"], "scheme": r["scheme"],
                           "oracle_gap_pct": float(r["gap_pct"]),
                           "real_vs_shuffled_gap_pct": float(res["gap_pct"]),
                           "p": float(res["p"]), "d": float(res["d"]),
                           "si_real": float(res["si_real"]), "si_shuf": float(res["si_shuf"])})
    return validation


def main():
    print("=" * 92)
    print("DOMAIN SCREENER: ranking (dataset x regime-scheme) by specialist-oracle gap")
    print("=" * 92)
    print("Looking for large gaps (>5%) = genuine regime structure to support the strong claim.\n")

    rows = []
    for label, csv, vcol, gcol, gval, rcol, native_causal in CANDIDATES:
        loaded = load_series(csv, vcol, gcol, gval, rcol)
        if loaded is None:
            print(f"  [skip] {label}: missing/short ({csv})")
            continue
        values, native = loaded

        schemes = {}
        # Only trust native labels that are known to be causal (clock-based).
        if native_causal and native is not None and len(set(native)) > 1:
            schemes["native"] = native
        for sname, fn in DATA_DRIVEN.items():
            schemes[sname] = fn(values)

        for sname, labels in schemes.items():
            res = oracle_gap(values, labels)
            rows.append({"dataset": label, "n": len(values), "scheme": sname, **res})

    rows.sort(key=lambda r: r["gap_pct"], reverse=True)

    print(f"{'dataset':<18}{'scheme':<12}{'n':>6}{'R':>4}{'winners':>9}{'gap%':>9}"
          f"{'gen_err':>11}{'spec_err':>11}")
    print("-" * 92)
    for r in rows:
        print(f"{r['dataset']:<18}{r['scheme']:<12}{r['n']:>6}{r['n_regimes']:>4}"
              f"{r['distinct_winners']:>9}{r['gap_pct']:>9.2f}"
              f"{r['generalist_err']:>11.3f}{r['specialist_err']:>11.3f}")

    print("\n" + "=" * 92)
    promising = [r for r in rows if r["gap_pct"] > 5.0 and r["distinct_winners"] > 1]
    validation = []
    if promising:
        print("PROMISING (gap > 5%, regimes disagree on best method):")
        for r in promising:
            print(f"  {r['dataset']} / {r['scheme']}: gap={r['gap_pct']:.2f}%  "
                  f"per-regime best = {r['best_per_regime']}")
        validation = validate_candidates(promising)
    else:
        print("NO candidate exceeds a 5% specialist-oracle gap. The strong claim has no")
        print("clean real-data support in this catalogue; lead with Route A (division of")
        print("labor) and the synthetic existence proof, and widen the data search.")

    out_dir = Path(__file__).parent.parent / "results" / "domain_screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    oracle_rows = [{k: (int(v) if isinstance(v, (np.integer,)) else
                        float(v) if isinstance(v, (np.floating,)) else v)
                    for k, v in r.items()} for r in rows]
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"experiment": "domain_screen_oracle_plus_validation",
                   "oracle_gaps": oracle_rows,
                   "coupled_real_vs_shuffled": validation}, f, indent=2)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
