#!/usr/bin/env python3
"""
Non-stationary regimes: does a population act as a distributed PERSISTENT MEMORY
that a capacity-bounded monolith cannot match?

exp_capacity_division.py showed specialization wins under bounded capacity via
spatial COVERAGE (no single agent can cover R>K regimes). This experiment isolates
a different, temporal mechanism: RETENTION under non-stationarity.

Setup. Regimes cycle: at each epoch only a sliding window of W of the R regimes is
active, so every regime is active for a while, goes dormant, then REACTIVATES.
Each agent has a bounded belief store of K regimes with LRU eviction (finite
memory): learning a new regime evicts the least-recently-used one.

  * MONOLITH (1 agent, K slots): must serve whatever regime is active now, so it
    evicts dormant regimes; when a regime reactivates after being evicted it must
    RELEARN from scratch (a fresh, uninformed store entry).
  * RANDOM-DIVERSITY (N=R agents, fixed K-regime niches): retains its assigned
    regimes across dormancy (control: distributed memory WITHOUT competition).
  * SPECIALIZED (N=R competitive agents): each agent settles on a niche and idles
    while that niche is dormant, so it RETAINS its niche's beliefs and is ready the
    moment the regime reactivates.

Prediction: under non-stationarity the specialized (and random) populations retain
distributed expertise and beat the monolith, with the gap concentrated in the
window right after each reactivation (the monolith's relearning penalty).
"""

import sys
import json
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._affinity_update import eg_eta_for_regimes


PROBE = 15  # steps after a reactivation counted as the "relearning window"


def make_nonstationary_stream(R=6, W=3, n_epochs=24, steps_per_epoch=80,
                              separation=1.0, noise=0.3, seed=0):
    """Sliding active-window regime-champion stream. Returns steps, regimes,
    methods, and a per-step boolean flag marking post-reactivation steps."""
    rng = np.random.default_rng(seed)
    regimes = [f"R{i}" for i in range(R)]
    methods = [f"M{i}" for i in range(R)]
    steps, reactivated = [], []
    prev_active = set()
    ever_active = set()
    for e in range(n_epochs):
        active = sorted({(e + j) % R for j in range(W)})
        fresh = set(active) - prev_active
        # A TRUE reactivation: active now, dormant last epoch, AND seen before (so
        # it was learned then forgotten). First-ever appearances are excluded so the
        # probe isolates relearning, not initial learning.
        true_reactivation = fresh & ever_active
        for s in range(steps_per_epoch):
            ri = active[rng.integers(len(active))]
            errs = {methods[mi]: abs(rng.normal(0.0, noise)) + (separation if mi != ri else 0.0)
                    for mi in range(R)}
            steps.append((regimes[ri], errs))
            reactivated.append((ri in true_reactivation) and s < PROBE)
        ever_active |= set(active)
        prev_active = set(active)
    return steps, regimes, methods, reactivated


# Soft-model knobs (interference account of forgetting; no discrete eviction).
_SOFT_EPS = 0.5       # min net pseudo-counts for a regime to count toward the footprint
_SOFT_INTER = 0.04    # base interference rate per foreign learning update


class StoreAgent:
    def __init__(self, regimes, methods, K, rng, fixed=None, cap_model="lru"):
        self.regimes, self.methods, self.K, self.rng = regimes, methods, K, rng
        self.cap_model = cap_model  # "lru" = hard slots + eviction; "soft" = graded interference
        self.store = OrderedDict()  # regime -> {method: [alpha, beta]}; insertion order = LRU
        self.fixed = fixed is not None
        # Heterogeneous init breaks symmetry so different agents are predisposed to
        # different niches (still emergent: the predisposition is random, not labelled).
        self.affinity = {r: 1.0 / len(regimes) + 0.05 * rng.random() for r in regimes}
        self._renorm()
        if self.fixed:
            for r in fixed:
                self.store[r] = {m: [1.0, 1.0] for m in methods}

    def _renorm(self):
        tot = sum(self.affinity.values())
        for r in self.affinity:
            self.affinity[r] /= tot

    def competent(self, r):
        return r in self.store

    def _strength(self, r):
        """Net pseudo-counts accumulated beyond the uninformed Beta(1,1) prior."""
        return sum(ab[0] + ab[1] - 2.0 for ab in self.store[r].values())

    def _interfere(self, current):
        """Soft capacity: when the agent's learned footprint exceeds K, every learning
        update bleeds knowledge out of the OTHER held regimes (proportional to overflow).
        Dormant regimes are never refreshed, so they decay toward the uninformed prior and
        are effectively forgotten -- the interference account of catastrophic forgetting,
        with NO discrete eviction rule."""
        footprint = sum(1 for r in self.store if self._strength(r) > _SOFT_EPS)
        overflow = max(0, footprint - self.K)
        if overflow == 0:
            return
        keep = 1.0 - _SOFT_INTER * overflow / max(footprint, 1)
        for r, methods in self.store.items():
            if r == current:
                continue
            for ab in methods.values():
                ab[0] = 1.0 + (ab[0] - 1.0) * keep
                ab[1] = 1.0 + (ab[1] - 1.0) * keep

    def ensure(self, r):
        """Make r learnable. Returns True if freshly (re)learned.
        LRU: hard K-slot store with least-recently-used eviction.
        Soft: lazily create the entry; capacity binds later via interference."""
        if r in self.store:
            if self.cap_model == "lru":
                self.store.move_to_end(r)
            return False
        if self.fixed:
            return False  # fixed agents never learn outside their assigned niches
        self.store[r] = {m: [1.0, 1.0] for m in self.methods}
        if self.cap_model == "lru" and len(self.store) > self.K:
            self.store.popitem(last=False)  # evict least-recently-used
        return True

    def select(self, r):
        if r in self.store:
            samples = {m: self.rng.beta(*self.store[r][m]) for m in self.methods}
            return max(samples, key=samples.get)
        return self.methods[self.rng.integers(len(self.methods))]  # uninformed

    def update_belief(self, r, m, success):
        if r in self.store:
            self.store[r][m][0 if success else 1] += 1.0
            if self.cap_model == "soft":
                self._interfere(r)

    def update_affinity(self, r, reward, eta):
        self.affinity[r] *= np.exp(eta * reward)
        self._renorm()


def run(steps, regimes, methods, K, mode, n_agents=None, lam=1.5, beta=1.0,
        gate_lr=0.2, eps_route=0.1, cap_model="lru", seed=0):
    rng = np.random.default_rng(seed)
    eta = eg_eta_for_regimes(len(regimes))
    R = len(regimes)
    N = R if n_agents is None else n_agents
    gate = {r: np.zeros(N) for r in regimes}  # MoE routing logits (regime -> experts)

    if mode == "monolith":
        agents = [StoreAgent(regimes, methods, K, rng, cap_model=cap_model)]
    elif mode == "random":
        agents = [StoreAgent(regimes, methods, K, rng, cap_model=cap_model,
                             fixed=[regimes[i] for i in rng.choice(R, size=K, replace=False)])
                  for _ in range(N)]
    else:
        agents = [StoreAgent(regimes, methods, K, rng, cap_model=cap_model) for _ in range(N)]

    err_all, err_react = [], []
    for (regime, errs), is_react in zip(steps, _react_iter(steps)):
        max_e = max(errs.values())
        span = (max_e - min(errs.values())) or 1.0
        quality, chosen = {}, {}
        for a in agents:
            m = a.select(regime)
            chosen[id(a)] = m
            quality[id(a)] = (max_e - errs[m]) / span

        if mode == "monolith":
            w = agents[0]
        elif mode == "moe":
            # Learned gating router: route the regime to an expert (epsilon-greedy),
            # train the gate on the routed expert's quality, and let the expert
            # learn/retain the regime in its bounded LRU store (capacity binds).
            e = int(rng.integers(N)) if rng.random() < eps_route else int(np.argmax(gate[regime]))
            expert = agents[e]
            expert.ensure(regime)
            expert.update_belief(regime, chosen[id(expert)], success=quality[id(expert)] >= 0.5)
            expert.update_affinity(regime, quality[id(expert)], eta)
            gate[regime][e] += gate_lr * (quality[id(expert)] - 0.5)
            w = None  # MoE handles its own learning; readout overridden below
        elif mode == "random":
            for a in agents:
                if a.competent(regime):
                    a.update_belief(regime, chosen[id(a)], success=quality[id(a)] >= 0.5)
            w = None
        elif mode == "diversity":
            # EOI-style learned diversity (no competition): all agents update affinity
            # with an intrinsic identifiability reward log p(i|regime); the identified
            # owner (argmax affinity) learns/retains the regime, giving distributed
            # memory via an explicit diversity objective rather than competition.
            denom = sum(a.affinity[regime] for a in agents) or 1.0
            owner = max(agents, key=lambda a: a.affinity[regime])
            for a in agents:
                p_i = a.affinity[regime] / denom
                a.update_affinity(regime, quality[id(a)] + beta * np.log(p_i + 1e-9), eta)
            owner.ensure(regime)
            owner.update_belief(regime, chosen[id(owner)], success=quality[id(owner)] >= 0.5)
            w = None
        else:
            # Competitive exclusion: favor the natural specialist for this regime
            # (+affinity[regime]) and penalize agents already committed elsewhere
            # (-(max affinity - affinity[regime])). This prevents one agent from
            # hoarding many niches during the cold start.
            scores = {}
            for a in agents:
                commit = max(a.affinity.values()) - a.affinity[regime]
                scores[id(a)] = quality[id(a)] + lam * a.affinity[regime] - lam * commit
            w = max(agents, key=lambda a: scores[id(a)])

        if w is not None:
            m = chosen[id(w)]  # method chosen before (re)learning; update it consistently
            w.ensure(regime)
            w.update_belief(regime, m, success=quality[id(w)] >= 0.5)
            w.update_affinity(regime, quality[id(w)], eta)

        if mode == "moe":
            served = agents[int(np.argmax(gate[regime]))]  # route via learned gate
        else:
            competent_here = [a for a in agents if a.competent(regime)]
            served = (max(competent_here, key=lambda a: a.affinity[regime]) if competent_here
                      else max(agents, key=lambda a: a.affinity[regime]))
        e = errs[served.select(regime)]
        err_all.append(e)
        if is_react:
            err_react.append(e)

    half = len(err_all) // 2
    return float(np.mean(err_all[half:])), float(np.mean(err_react) if err_react else np.nan)


# reactivation flags are produced alongside the stream; cache them per-stream id.
_REACT_CACHE = {}


def _react_iter(steps):
    return _REACT_CACHE[id(steps)]


def trials(steps, regimes, methods, K, mode, n_trials=20, **kw):
    out = [run(steps, regimes, methods, K, mode, seed=s, **kw) for s in range(n_trials)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def _sem(arr):
    """Standard error of the mean (for 95% CI = 1.96*SEM error bars)."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


def main():
    print("#" * 90)
    print("# NON-STATIONARY REGIMES: population as distributed persistent memory vs a")
    print("# capacity-bounded monolith that must evict and relearn dormant regimes")
    print("#" * 90)
    R, W = 6, 3
    steps, regimes, methods, react = make_nonstationary_stream(R=R, W=W, seed=0)
    _REACT_CACHE[id(steps)] = react
    print(f"\nStream: R={R} regimes, sliding window W={W} active at a time, "
          f"{len(steps)} steps; each regime dormant {R - W} epochs then reactivates.")
    print(f"Post-reactivation probe window: first {PROBE} steps after each reactivation.\n")

    print(f"{'K':>3}{'monolith':>17}{'random-div':>17}{'EOI-div':>17}{'MoE-gate':>17}{'specialized':>17}")
    print(f"{'':>3}{'over / react':>17}{'over / react':>17}{'over / react':>17}{'over / react':>17}{'over / react':>17}")
    print("-" * 105)
    rows = []
    for K in range(1, R + 1):
        mo, mr = trials(steps, regimes, methods, K, "monolith")
        ro, rr = trials(steps, regimes, methods, K, "random")
        do, dr = trials(steps, regimes, methods, K, "diversity")
        go, gr = trials(steps, regimes, methods, K, "moe")
        so, sr = trials(steps, regimes, methods, K, "specialized")
        _t, p_all = stats.ttest_ind(so, mo, alternative="less")
        _t, p_re = stats.ttest_ind(sr, mr, alternative="less")
        _t, p_div_re = stats.ttest_ind(sr, dr, alternative="less")
        _t, p_moe_re = stats.ttest_ind(sr, gr, alternative="less")
        rows.append({"K": K,
                     "monolith_overall": float(mo.mean()), "monolith_react": float(mr.mean()),
                     "random_overall": float(ro.mean()), "random_react": float(rr.mean()),
                     "diversity_overall": float(do.mean()), "diversity_react": float(dr.mean()),
                     "moe_overall": float(go.mean()), "moe_react": float(gr.mean()),
                     "specialized_overall": float(so.mean()), "specialized_react": float(sr.mean()),
                     "spec_vs_mono_overall_p": float(p_all), "spec_vs_mono_react_p": float(p_re),
                     "spec_vs_diversity_react_p": float(p_div_re),
                     "spec_vs_moe_react_p": float(p_moe_re),
                     "monolith_overall_sem": _sem(mo), "monolith_react_sem": _sem(mr),
                     "random_overall_sem": _sem(ro), "random_react_sem": _sem(rr),
                     "diversity_overall_sem": _sem(do), "diversity_react_sem": _sem(dr),
                     "moe_overall_sem": _sem(go), "moe_react_sem": _sem(gr),
                     "specialized_overall_sem": _sem(so), "specialized_react_sem": _sem(sr)})
        print(f"{K:>3}"
              f"{f'{mo.mean():.3f}/{mr.mean():.3f}':>17}"
              f"{f'{ro.mean():.3f}/{rr.mean():.3f}':>17}"
              f"{f'{do.mean():.3f}/{dr.mean():.3f}':>17}"
              f"{f'{go.mean():.3f}/{gr.mean():.3f}':>17}"
              f"{f'{so.mean():.3f}/{sr.mean():.3f}':>17}")
    print("-" * 105)
    print("EOI-div = explicit learned-diversity objective; MoE-gate = learned routing of capacity-K experts.")

    K = 3
    r3 = next(r for r in rows if r["K"] == K)
    print(f"\nAt K={K} (< R={R}): specialized vs monolith")
    print(f"  overall error:           {r3['specialized_overall']:.3f} vs {r3['monolith_overall']:.3f}  "
          f"({(r3['monolith_overall']-r3['specialized_overall'])/r3['monolith_overall']*100:+.1f}%, "
          f"p={r3['spec_vs_mono_overall_p']:.2g})")
    print(f"  post-reactivation error: {r3['specialized_react']:.3f} vs {r3['monolith_react']:.3f}  "
          f"({(r3['monolith_react']-r3['specialized_react'])/r3['monolith_react']*100:+.1f}%, "
          f"p={r3['spec_vs_mono_react_p']:.2g})")
    print("\nThe population retains dormant-niche expertise; the monolith relearns each")
    print("time a regime reactivates -> the gap is largest in the post-reactivation window.")

    save_and_plot(rows, R, W, Path(__file__).parent.parent / "results" / "nonstationary_capacity")

    if "--soft" in sys.argv:
        soft_robustness(R, W)


def soft_robustness(R, W):
    """P3 robustness: re-run the memory result under a SOFT (interference) capacity model
    instead of hard LRU eviction, to show catastrophic forgetting is not an artifact of the
    discrete eviction rule. Forgetting here is graded belief decay under capacity overflow."""
    print("\n" + "#" * 90)
    print("# ROBUSTNESS: soft interference capacity model (graded decay, NO discrete eviction)")
    print("#" * 90)
    steps, regimes, methods, react = make_nonstationary_stream(R=R, W=W, seed=0)
    _REACT_CACHE[id(steps)] = react
    print(f"\n{'K':>3}{'monolith':>17}{'random-div':>17}{'EOI-div':>17}{'MoE-gate':>17}{'specialized':>17}")
    print(f"{'':>3}{'over / react':>17}{'over / react':>17}{'over / react':>17}{'over / react':>17}{'over / react':>17}")
    print("-" * 105)
    rows = []
    for K in range(1, R + 1):
        mo, mr = trials(steps, regimes, methods, K, "monolith", cap_model="soft")
        ro, rr = trials(steps, regimes, methods, K, "random", cap_model="soft")
        do, dr = trials(steps, regimes, methods, K, "diversity", cap_model="soft")
        go, gr = trials(steps, regimes, methods, K, "moe", cap_model="soft")
        so, sr = trials(steps, regimes, methods, K, "specialized", cap_model="soft")
        _t, p_re = stats.ttest_ind(sr, mr, alternative="less")
        _t, p_moe_re = stats.ttest_ind(sr, gr, alternative="less")
        rows.append({"K": K,
                     "monolith_overall": float(mo.mean()), "monolith_react": float(mr.mean()),
                     "random_overall": float(ro.mean()), "random_react": float(rr.mean()),
                     "diversity_overall": float(do.mean()), "diversity_react": float(dr.mean()),
                     "moe_overall": float(go.mean()), "moe_react": float(gr.mean()),
                     "specialized_overall": float(so.mean()), "specialized_react": float(sr.mean()),
                     "spec_vs_mono_react_p": float(p_re), "spec_vs_moe_react_p": float(p_moe_re),
                     "monolith_overall_sem": _sem(mo), "monolith_react_sem": _sem(mr),
                     "random_overall_sem": _sem(ro), "random_react_sem": _sem(rr),
                     "diversity_overall_sem": _sem(do), "diversity_react_sem": _sem(dr),
                     "moe_overall_sem": _sem(go), "moe_react_sem": _sem(gr),
                     "specialized_overall_sem": _sem(so), "specialized_react_sem": _sem(sr)})
        print(f"{K:>3}"
              f"{f'{mo.mean():.3f}/{mr.mean():.3f}':>17}"
              f"{f'{ro.mean():.3f}/{rr.mean():.3f}':>17}"
              f"{f'{do.mean():.3f}/{dr.mean():.3f}':>17}"
              f"{f'{go.mean():.3f}/{gr.mean():.3f}':>17}"
              f"{f'{so.mean():.3f}/{sr.mean():.3f}':>17}")
    print("-" * 105)
    for K in (1, 3):
        r = next(x for x in rows if x["K"] == K)
        print(f"K={K}: post-react specialized {r['specialized_react']:.3f} vs monolith "
              f"{r['monolith_react']:.3f} ({(r['monolith_react']-r['specialized_react'])/r['monolith_react']*100:+.1f}%, "
              f"p={r['spec_vs_mono_react_p']:.2g}); vs MoE router {r['moe_react']:.3f} "
              f"({(r['moe_react']-r['specialized_react'])/r['moe_react']*100:+.1f}%, p={r['spec_vs_moe_react_p']:.2g})")
    save_and_plot(rows, R, W, Path(__file__).parent.parent / "results" / "nonstationary_capacity",
                  prefix="nonstationary_soft", json_name="results_soft.json",
                  paper_fig_name="fig8_nonstationary_soft.pdf",
                  suptitle=f"Robustness: soft interference capacity model (R={R}, W={W})")


def save_and_plot(rows, R, W, out_dir, prefix="nonstationary_sweep",
                  json_name="results.json", paper_fig_name="fig7_nonstationary.pdf",
                  suptitle=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / json_name, "w", encoding="utf-8") as f:
        json.dump({"R": R, "W": W, "probe": PROBE, "rows": rows}, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Results saved to {out_dir} (matplotlib unavailable; skipped figure).")
        return

    ks = [r["K"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (suffix, title) in zip(axes, [("overall", "Overall error"),
                                          ("react", "Post-reactivation error")]):
        for key, style, lab, col in [
                ("monolith", "o-", "monolith (cap. K)", "#d62728"),
                ("random", "s--", "random diversity", "#7f7f7f"),
                ("diversity", "D:", "EOI diversity", "#2ca02c"),
                ("moe", "v-.", "MoE router", "#9467bd"),
                ("specialized", "^-", "specialized (ours)", "#1f77b4")]:
            ax.errorbar(ks, [r.get(f"{key}_{suffix}", np.nan) for r in rows],
                        yerr=[1.96 * r.get(f"{key}_{suffix}_sem", 0.0) for r in rows],
                        fmt=style, label=lab, color=col, capsize=2, markersize=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("per-agent memory capacity K")
        ax.set_ylabel("prediction error")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(suptitle or
                 f"Non-stationary regimes (R={R}, W={W}): population as distributed persistent memory",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / f"{prefix}.png", dpi=150)
    paper_fig = Path(__file__).parent.parent / "paper" / "figures" / paper_fig_name
    if paper_fig.parent.exists():
        fig.savefig(paper_fig)
    print(f"Results + figure saved to {out_dir}")


if __name__ == "__main__":
    main()
