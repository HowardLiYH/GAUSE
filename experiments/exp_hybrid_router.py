#!/usr/bin/env python3
"""
Hybrid arm: a REWARD-DRIVEN router augmented with a memory-preservation
(capacity-reservation) term. Does it retain dormant regimes?

Motivation. The non-stationarity result (exp_nonstationary_capacity.py) shows the
vanilla learned MoE router FORGETS dormant regimes: a reward-driven gate gets no
signal from a dormant regime, so it reuses the expert and relearns on reactivation.
A natural objection is that this is a strawman --- a real systems engineer would add
an explicit reservation/load-balancing term. This experiment adds exactly that and
asks what it buys, completing the design space the reward-independence principle
predicts.

The hybrid arm ("moe_preserve") keeps the router's reward-driven ROUTING unchanged
(epsilon-greedy gate trained on routed-expert quality) but adds a PROTECTION term:
once the gate has committed a regime r to an expert e (e is the argmax owner of r),
e PINS r's beliefs --- r is never evicted and never decayed by interference, even
while dormant. Capacity is thus RESERVED for assigned regimes regardless of current
activity.

Prediction (reward-independence principle, scoped form). Retention is governed by
whether the PROTECTION of capacity is reward-independent, not by whether ROUTING is.
The vanilla router forgets (protection chases reward); the hybrid router retains
(protection is reward-independent once the gate commits), matching GAUSE --- which
gets the same reward-independent protection for free, with no reservation bookkeeping.
This shows reward-independence is the operative property and is achievable several
ways; competition is simply the cheapest.
"""

import sys
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._affinity_update import eg_eta_for_regimes
from experiments.exp_nonstationary_capacity import (
    make_nonstationary_stream, StoreAgent, PROBE, _sem, _REACT_CACHE, _react_iter,
)

# Gate-commitment threshold: an expert "owns" (and protects) regime r once the
# gate's logit for (r -> e) exceeds this and e is the argmax expert for r.
_COMMIT = 0.15


class PreservingStoreAgent(StoreAgent):
    """StoreAgent that pins the beliefs of regimes it has been assigned to protect.
    Protected regimes are exempt from LRU eviction and from soft interference."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protected = set()

    def set_protected(self, regimes_set):
        self.protected = set(regimes_set)

    def ensure(self, r):
        if r in self.store:
            if self.cap_model == "lru":
                self.store.move_to_end(r)
            return False
        if self.fixed:
            return False
        self.store[r] = {m: [1.0, 1.0] for m in self.methods}
        if self.cap_model == "lru" and len(self.store) > self.K:
            victim = next((c for c in self.store
                           if c != r and c not in self.protected), None)
            if victim is None:  # all others protected: capacity is hard, evict oldest non-r
                victim = next((c for c in self.store if c != r), None)
            if victim is not None:
                del self.store[victim]
        return True

    def _interfere(self, current):
        footprint = sum(1 for r in self.store if self._strength(r) > 0.5)
        overflow = max(0, footprint - self.K)
        if overflow == 0:
            return
        keep = 1.0 - 0.04 * overflow / max(footprint, 1)
        for r, methods in self.store.items():
            if r == current or r in self.protected:  # protected niches do not decay
                continue
            for ab in methods.values():
                ab[0] = 1.0 + (ab[0] - 1.0) * keep
                ab[1] = 1.0 + (ab[1] - 1.0) * keep


def run(steps, regimes, methods, K, mode, lam=1.5, gate_lr=0.2, eps_route=0.1,
        cap_model="lru", seed=0):
    rng = np.random.default_rng(seed)
    eta = eg_eta_for_regimes(len(regimes))
    R = len(regimes)
    N = R
    gate = {r: np.zeros(N) for r in regimes}

    AgentCls = PreservingStoreAgent if mode == "moe_preserve" else StoreAgent
    if mode == "monolith":
        agents = [AgentCls(regimes, methods, K, rng, cap_model=cap_model)]
    else:
        agents = [AgentCls(regimes, methods, K, rng, cap_model=cap_model) for _ in range(N)]

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
        elif mode in ("moe", "moe_preserve"):
            e = int(rng.integers(N)) if rng.random() < eps_route else int(np.argmax(gate[regime]))
            expert = agents[e]
            expert.ensure(regime)
            expert.update_belief(regime, chosen[id(expert)], success=quality[id(expert)] >= 0.5)
            expert.update_affinity(regime, quality[id(expert)], eta)
            gate[regime][e] += gate_lr * (quality[id(expert)] - 0.5)
            if mode == "moe_preserve":
                # Reservation term: pin every (expert, regime) the gate has committed to.
                for rr in regimes:
                    owner = int(np.argmax(gate[rr]))
                    if gate[rr][owner] > _COMMIT and rr in agents[owner].store:
                        protset = getattr(agents[owner], "protected", set()) | {rr}
                        agents[owner].set_protected(protset)
            w = None
        else:  # specialized (GAUSE)
            scores = {}
            for a in agents:
                commit = max(a.affinity.values()) - a.affinity[regime]
                scores[id(a)] = quality[id(a)] + lam * a.affinity[regime] - lam * commit
            w = max(agents, key=lambda a: scores[id(a)])

        if w is not None:
            m = chosen[id(w)]
            w.ensure(regime)
            w.update_belief(regime, m, success=quality[id(w)] >= 0.5)
            w.update_affinity(regime, quality[id(w)], eta)

        if mode in ("moe", "moe_preserve"):
            served = agents[int(np.argmax(gate[regime]))]
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


def trials(steps, regimes, methods, K, mode, n_trials=20, **kw):
    out = [run(steps, regimes, methods, K, mode, seed=s, **kw) for s in range(n_trials)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def main():
    print("#" * 90)
    print("# HYBRID ARM: reward-driven router + memory-preservation (reservation) term")
    print("#" * 90)
    R, W = 6, 3
    steps, regimes, methods, react = make_nonstationary_stream(R=R, W=W, seed=0)
    _REACT_CACHE[id(steps)] = react
    print(f"\nStream: R={R}, W={W}, {len(steps)} steps; probe={PROBE}.\n")
    print(f"{'K':>3}{'monolith':>16}{'MoE (vanilla)':>16}{'MoE+preserve':>16}{'GAUSE':>16}")
    print(f"{'':>3}{'over/react':>16}{'over/react':>16}{'over/react':>16}{'over/react':>16}")
    print("-" * 80)
    rows = []
    for K in range(1, R + 1):
        mo, mr = trials(steps, regimes, methods, K, "monolith")
        go, gr = trials(steps, regimes, methods, K, "moe")
        ho, hr = trials(steps, regimes, methods, K, "moe_preserve")
        so, sr = trials(steps, regimes, methods, K, "specialized")
        _t, p_hyb_vs_moe = stats.ttest_ind(hr, gr, alternative="less")
        _t, p_hyb_vs_spec = stats.ttest_ind(hr, sr, alternative="two-sided")
        rows.append({"K": K,
                     "monolith_overall": float(mo.mean()), "monolith_react": float(mr.mean()),
                     "moe_overall": float(go.mean()), "moe_react": float(gr.mean()),
                     "moe_preserve_overall": float(ho.mean()), "moe_preserve_react": float(hr.mean()),
                     "specialized_overall": float(so.mean()), "specialized_react": float(sr.mean()),
                     "hybrid_vs_moe_react_p": float(p_hyb_vs_moe),
                     "hybrid_vs_spec_react_p": float(p_hyb_vs_spec),
                     "monolith_overall_sem": _sem(mo), "monolith_react_sem": _sem(mr),
                     "moe_overall_sem": _sem(go), "moe_react_sem": _sem(gr),
                     "moe_preserve_overall_sem": _sem(ho), "moe_preserve_react_sem": _sem(hr),
                     "specialized_overall_sem": _sem(so), "specialized_react_sem": _sem(sr)})
        print(f"{K:>3}"
              f"{f'{mo.mean():.3f}/{mr.mean():.3f}':>16}"
              f"{f'{go.mean():.3f}/{gr.mean():.3f}':>16}"
              f"{f'{ho.mean():.3f}/{hr.mean():.3f}':>16}"
              f"{f'{so.mean():.3f}/{sr.mean():.3f}':>16}")
    print("-" * 80)
    for K in (1, 3):
        r = next(x for x in rows if x["K"] == K)
        red_hyb = (r["moe_react"] - r["moe_preserve_react"]) / r["moe_react"] * 100
        print(f"K={K}: post-react  MoE(vanilla)={r['moe_react']:.3f}  "
              f"MoE+preserve={r['moe_preserve_react']:.3f} ({red_hyb:+.1f}% vs vanilla, "
              f"p={r['hybrid_vs_moe_react_p']:.2g})  GAUSE={r['specialized_react']:.3f} "
              f"(hybrid vs GAUSE p={r['hybrid_vs_spec_react_p']:.2g})")
    print("\nReading: the reservation term recovers most of the router's lost retention while")
    print("keeping reward-driven routing -> reward-independence of PROTECTION is the operative")
    print("property; GAUSE obtains it for free with no reservation bookkeeping.")

    save_and_plot(rows, R, W, Path(__file__).parent.parent / "results" / "hybrid_router")


def save_and_plot(rows, R, W, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"R": R, "W": W, "probe": PROBE, "commit_threshold": _COMMIT, "rows": rows},
                  f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Results saved to {out_dir} (matplotlib unavailable).")
        return
    ks = [r["K"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (suffix, title) in zip(axes, [("overall", "Overall error"),
                                          ("react", "Post-reactivation error")]):
        for key, style, lab, col in [
                ("monolith", "o-", "monolith (cap. K)", "#d62728"),
                ("moe", "v-.", "MoE router (vanilla)", "#9467bd"),
                ("moe_preserve", "P--", "MoE + preservation (hybrid)", "#ff7f0e"),
                ("specialized", "^-", "GAUSE (ours)", "#1f77b4")]:
            ax.errorbar(ks, [r.get(f"{key}_{suffix}", np.nan) for r in rows],
                        yerr=[1.96 * r.get(f"{key}_{suffix}_sem", 0.0) for r in rows],
                        fmt=style, label=lab, color=col, capsize=2, markersize=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("per-agent memory capacity K")
        ax.set_ylabel("prediction error")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Hybrid router (reward-driven routing + reservation), R={R}, W={W}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "hybrid_router.png", dpi=150)
    paper_fig = Path(__file__).parent.parent / "paper" / "figures" / "fig9_hybrid_router.pdf"
    if paper_fig.parent.exists():
        fig.savefig(paper_fig)
    print(f"Results + figure saved to {out_dir}")


if __name__ == "__main__":
    main()
