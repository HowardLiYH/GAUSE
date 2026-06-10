#!/usr/bin/env python3
"""
Intra-regime concept drift: the blind spot of win-only identity updates.

The EG affinity update moves a learner's identity only on WINS (positive evidence of
fit). That asymmetry is what makes converged identities stable through DORMANCY (a
regime that pauses and returns UNCHANGED) -- the property the non-stationarity
experiment exploits. This experiment tests the other case the asymmetry creates:
INTRA-REGIME DRIFT, where a regime returns CHANGED (its optimal method has drifted
while its specialist idled). A converged specialist holds the niche by identity, so
it keeps serving the regime with stale beliefs and does not spontaneously relinquish
it -- the "stale specialist" risk.

Stream. Regimes cycle (sliding window W of R, dormant then reactivating), and on
every reactivation a regime's CHAMPION method shifts by one (concept drift). The
probe window is the first PROBE steps after each DRIFTED reactivation.

Arms.
  * monolith        : single capacity-K agent (reference).
  * gause           : standard winner-take-all + EG affinity (no drift handling). The
                      incumbent owner keeps winning via its affinity bonus and must
                      slowly overwrite its stale beliefs.
  * gause_staleness : same, plus a STALENESS TRIGGER -- when the served owner's recent
                      realized quality on its niche collapses (confidently wrong, i.e.
                      drift rather than noise), reset that regime's beliefs to the prior
                      and soften the owner's affinity, re-opening competition / enabling
                      fast relearning.

Prediction. Standard GAUSE pays an elevated post-drift error (stale specialist);
the staleness trigger recovers most of it. This both demonstrates the limitation
honestly and shows a concrete, cheap remedy.
"""

import sys
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._affinity_update import eg_eta_for_regimes

PROBE = 15
_EMA = 0.3            # smoothing for the realized-quality tracker
_STALE_Q = 0.45       # quality EMA below this on one's own niche => candidate stale
_STALE_MIN_COUNT = 6  # require confident beliefs (drift, not cold start / noise)


def make_drift_stream(R=6, W=3, n_epochs=24, steps_per_epoch=80,
                      separation=1.0, noise=0.3, seed=0):
    """Sliding-window regime stream whose per-regime CHAMPION method drifts by +1 on
    every reactivation. Returns steps, regimes, methods, and a per-step flag marking
    post-drift probe steps."""
    rng = np.random.default_rng(seed)
    regimes = [f"R{i}" for i in range(R)]
    methods = [f"M{i}" for i in range(R)]
    champion = {i: i for i in range(R)}     # regime index -> current best method index
    steps, post_drift = [], []
    prev_active, ever_active = set(), set()
    for e in range(n_epochs):
        active = sorted({(e + j) % R for j in range(W)})
        fresh = set(active) - prev_active
        reactivated = fresh & ever_active
        drifted_now = set()
        for ri in sorted(reactivated):       # drift each reactivating regime's champion
            champion[ri] = (champion[ri] + 1) % R
            drifted_now.add(ri)
        for s in range(steps_per_epoch):
            ri = active[rng.integers(len(active))]
            ch = champion[ri]
            errs = {methods[mi]: abs(rng.normal(0.0, noise)) + (0.0 if mi == ch else separation)
                    for mi in range(R)}
            steps.append((regimes[ri], errs))
            post_drift.append((ri in drifted_now) and s < PROBE)
        ever_active |= set(active)
        prev_active = set(active)
    return steps, regimes, methods, post_drift


class DriftAgent:
    def __init__(self, regimes, methods, K, rng):
        self.regimes, self.methods, self.K, self.rng = regimes, methods, K, rng
        self.store = OrderedDict()                       # regime -> {method:[a,b]}
        self.affinity = {r: 1.0 / len(regimes) + 0.05 * rng.random() for r in regimes}
        self._renorm()
        self.q_ema = defaultdict(lambda: 0.5)            # (regime) -> realized quality EMA

    def _renorm(self):
        tot = sum(self.affinity.values())
        for r in self.affinity:
            self.affinity[r] /= tot

    def competent(self, r):
        return r in self.store

    def ensure(self, r):
        if r in self.store:
            self.store.move_to_end(r)
            return
        self.store[r] = {m: [1.0, 1.0] for m in self.methods}
        if len(self.store) > self.K:
            self.store.popitem(last=False)

    def select(self, r):
        if r in self.store:
            samples = {m: self.rng.beta(*self.store[r][m]) for m in self.methods}
            return max(samples, key=samples.get)
        return self.methods[self.rng.integers(len(self.methods))]

    def belief_count(self, r):
        if r not in self.store:
            return 0.0
        return sum(a + b - 2.0 for a, b in self.store[r].values())

    def update_belief(self, r, m, success):
        if r in self.store:
            self.store[r][m][0 if success else 1] += 1.0

    def update_affinity(self, r, reward, eta):
        self.affinity[r] *= np.exp(eta * reward)
        self._renorm()

    def note_quality(self, r, q):
        self.q_ema[r] = (1 - _EMA) * self.q_ema[r] + _EMA * q

    def reset_regime(self, r):
        """Staleness response: drop stale beliefs and soften identity so competition re-opens."""
        if r in self.store:
            self.store[r] = {m: [1.0, 1.0] for m in self.methods}
        self.q_ema[r] = 0.5
        # soften affinity toward uniform so a better-suited agent can contest the niche
        u = 1.0 / len(self.regimes)
        self.affinity[r] = 0.5 * self.affinity[r] + 0.5 * u
        self._renorm()


def run(steps, post_drift, regimes, methods, K, mode, lam=1.5, seed=0):
    rng = np.random.default_rng(seed)
    eta = eg_eta_for_regimes(len(regimes))
    R = len(regimes)
    N = 1 if mode == "monolith" else R
    agents = [DriftAgent(regimes, methods, K, rng) for _ in range(N)]

    err_all, err_drift = [], []
    for (regime, errs), is_pd in zip(steps, post_drift):
        max_e = max(errs.values())
        span = (max_e - min(errs.values())) or 1.0
        quality, chosen = {}, {}
        for a in agents:
            m = a.select(regime)
            chosen[id(a)] = m
            quality[id(a)] = (max_e - errs[m]) / span

        if mode == "monolith":
            w = agents[0]
        else:
            scores = {}
            for a in agents:
                commit = max(a.affinity.values()) - a.affinity[regime]
                scores[id(a)] = quality[id(a)] + lam * a.affinity[regime] - lam * commit
            w = max(agents, key=lambda a: scores[id(a)])

        m = chosen[id(w)]
        w.ensure(regime)
        w.update_belief(regime, m, success=quality[id(w)] >= 0.5)
        w.update_affinity(regime, quality[id(w)], eta)
        w.note_quality(regime, quality[id(w)])

        # Staleness trigger: the owner is confidently underperforming on its OWN niche.
        if mode == "gause_staleness":
            if (w.q_ema[regime] < _STALE_Q and w.belief_count(regime) > _STALE_MIN_COUNT
                    and w.affinity[regime] >= max(w.affinity.values()) - 1e-9):
                w.reset_regime(regime)

        # readout: the affinity-owner competent here serves the regime
        competent_here = [a for a in agents if a.competent(regime)]
        served = (max(competent_here, key=lambda a: a.affinity[regime]) if competent_here
                  else max(agents, key=lambda a: a.affinity[regime]))
        e = errs[served.select(regime)]
        err_all.append(e)
        if is_pd:
            err_drift.append(e)

    half = len(err_all) // 2
    return float(np.mean(err_all[half:])), float(np.mean(err_drift) if err_drift else np.nan)


def trials(steps, post_drift, regimes, methods, K, mode, n_trials=20, **kw):
    out = [run(steps, post_drift, regimes, methods, K, mode, seed=s, **kw) for s in range(n_trials)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def _sem(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


def main():
    print("#" * 90)
    print("# INTRA-REGIME CONCEPT DRIFT: stale specialists and a staleness-trigger remedy")
    print("#" * 90)
    R, W = 6, 3
    steps, regimes, methods, post_drift = make_drift_stream(R=R, W=W, seed=0)
    n_drift = sum(post_drift)
    print(f"\nStream: R={R}, W={W}, {len(steps)} steps; champion method drifts +1 on each")
    print(f"reactivation; post-drift probe steps flagged = {n_drift}.\n")
    print(f"{'K':>3}{'monolith':>18}{'GAUSE':>18}{'GAUSE+staleness':>18}")
    print(f"{'':>3}{'over/drift':>18}{'over/drift':>18}{'over/drift':>18}")
    print("-" * 60)
    rows = []
    for K in range(1, R + 1):
        mo, md = trials(steps, post_drift, regimes, methods, K, "monolith")
        go, gd = trials(steps, post_drift, regimes, methods, K, "gause")
        so, sd = trials(steps, post_drift, regimes, methods, K, "gause_staleness")
        _t, p_stale = stats.ttest_ind(sd, gd, alternative="less")
        rows.append({"K": K,
                     "monolith_overall": float(mo.mean()), "monolith_drift": float(md.mean()),
                     "gause_overall": float(go.mean()), "gause_drift": float(gd.mean()),
                     "staleness_overall": float(so.mean()), "staleness_drift": float(sd.mean()),
                     "staleness_vs_gause_drift_p": float(p_stale),
                     "monolith_overall_sem": _sem(mo), "monolith_drift_sem": _sem(md),
                     "gause_overall_sem": _sem(go), "gause_drift_sem": _sem(gd),
                     "staleness_overall_sem": _sem(so), "staleness_drift_sem": _sem(sd)})
        print(f"{K:>3}"
              f"{f'{mo.mean():.3f}/{md.mean():.3f}':>18}"
              f"{f'{go.mean():.3f}/{gd.mean():.3f}':>18}"
              f"{f'{so.mean():.3f}/{sd.mean():.3f}':>18}")
    print("-" * 60)
    for K in (1, 3):
        r = next(x for x in rows if x["K"] == K)
        red = (r["gause_drift"] - r["staleness_drift"]) / r["gause_drift"] * 100
        print(f"K={K}: post-drift  GAUSE={r['gause_drift']:.3f}  "
              f"GAUSE+staleness={r['staleness_drift']:.3f} ({red:+.1f}%, "
              f"p={r['staleness_vs_gause_drift_p']:.2g})")
    print("\nReading: standard GAUSE pays a post-drift penalty (stale specialist holds the")
    print("niche with outdated beliefs); a cheap staleness trigger that resets confidently-")
    print("wrong niches recovers most of it -- confirming the limitation and a remedy.")

    save_and_plot(rows, R, W, Path(__file__).parent.parent / "results" / "intra_regime_drift")


def save_and_plot(rows, R, W, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"R": R, "W": W, "probe": PROBE, "stale_q": _STALE_Q, "rows": rows}, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Results saved to {out_dir} (matplotlib unavailable).")
        return
    ks = [r["K"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    # Shared y-range across panels so the narrow post-drift spread is shown in the
    # same visual scale as the overall errors (no exaggerated zoom).
    all_vals = [r[f"{key}_{suffix}"] for r in rows
                for key in ("monolith", "gause", "staleness")
                for suffix in ("overall", "drift")]
    y_lo, y_hi = min(all_vals), max(all_vals)
    pad = 0.05 * (y_hi - y_lo)
    for ax, (suffix, title) in zip(axes, [("overall", "Overall error"),
                                          ("drift", "Post-drift error")]):
        for key, style, lab, col in [
                ("monolith", "o-", "monolith (cap. K)", "#d62728"),
                ("gause", "^-", "GAUSE (no drift handling)", "#1f77b4"),
                ("staleness", "*--", "GAUSE + staleness trigger", "#2ca02c")]:
            ax.errorbar(ks, [r.get(f"{key}_{suffix}", np.nan) for r in rows],
                        yerr=[1.96 * r.get(f"{key}_{suffix}_sem", 0.0) for r in rows],
                        fmt=style, label=lab, color=col, capsize=2, markersize=6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("per-agent memory capacity K")
        ax.set_ylabel("prediction error")
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Intra-regime concept drift (R={R}, W={W}): stale specialists and a remedy",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "intra_regime_drift.png", dpi=150)
    paper_fig = Path(__file__).parent.parent / "paper" / "figures" / "fig10_intra_regime_drift.pdf"
    if paper_fig.parent.exists():
        fig.savefig(paper_fig)
    print(f"Results + figure saved to {out_dir}")


if __name__ == "__main__":
    main()
