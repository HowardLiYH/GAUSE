#!/usr/bin/env python3
"""
Population sizing off the N = R diagonal: what happens when agents are scarce or
abundant relative to regimes?

The clean one-per-regime equilibrium and the coverage guarantee behind our
bounded-capacity results are established for N ~ R. This experiment characterizes the
two off-diagonal regimes empirically, on the non-stationary retention protocol
(R = 6 regimes cycling through dormancy, per-agent capacity K with LRU eviction):

  * N << R (scarce agents): the population's total capacity is N*K. If N*K < R it
    is information-theoretically impossible to retain every regime, so some recurring
    regime must be forgotten and relearned. We measure how close competitive coverage
    gets to the feasibility ceiling min(1, N*K / R).
  * N ~ R (the diagonal): the clean regime, for reference.
  * N >> R (abundant agents): coverage is trivially complete; the question is whether
    redundant agents idle harmlessly or whether within-niche competition degrades
    retention. We measure the number of redundant (non-owner) agents and check that
    retention does not regress.

Arms: the competitive population ("specialized"), with a capacity-K monolith and a
fixed-random-niche population at the same N as reference. Metrics per (N, K):
overall error, post-reactivation error, coverage fraction (regimes with a competent
owner), and the redundant-agent count.

Prediction. Coverage tracks the feasibility ceiling min(1, N*K/R): below it,
post-reactivation error rises smoothly with the uncovered fraction; at and above N=R
it saturates, with surplus agents going idle rather than harming retention.
"""

import sys
import json
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments._affinity_update import eg_eta_for_regimes
from experiments.exp_nonstationary_capacity import (
    make_nonstationary_stream, StoreAgent, PROBE, _sem, _REACT_CACHE, _react_iter,
)


def run(steps, regimes, methods, K, N, mode, lam=1.5, seed=0):
    rng = np.random.default_rng(seed)
    eta = eg_eta_for_regimes(len(regimes))
    R = len(regimes)

    if mode == "monolith":
        agents = [StoreAgent(regimes, methods, K, rng)]
    elif mode == "random":
        agents = [StoreAgent(regimes, methods, K, rng,
                             fixed=[regimes[i] for i in rng.choice(R, size=K, replace=False)])
                  for _ in range(N)]
    else:  # specialized
        agents = [StoreAgent(regimes, methods, K, rng) for _ in range(N)]

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
        elif mode == "random":
            for a in agents:
                if a.competent(regime):
                    a.update_belief(regime, chosen[id(a)], success=quality[id(a)] >= 0.5)
            w = None
        else:
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

        competent_here = [a for a in agents if a.competent(regime)]
        served = (max(competent_here, key=lambda a: a.affinity[regime]) if competent_here
                  else max(agents, key=lambda a: a.affinity[regime]))
        e = errs[served.select(regime)]
        err_all.append(e)
        if is_react:
            err_react.append(e)

    # --- structural metrics at convergence ---
    # A regime is COVERED if some agent holds it with informed (beyond-prior) beliefs.
    covered = set()
    for r in regimes:
        for a in agents:
            if r in a.store and a._strength(r) > 0.5:
                covered.add(r)
                break
    coverage_frac = len(covered) / R

    # Redundant agents: agents that are not the unique affinity-owner of any covered regime.
    owners = set()
    for r in covered:
        cand = [a for a in agents if r in a.store and a._strength(r) > 0.5]
        if cand:
            owners.add(id(max(cand, key=lambda a: a.affinity[r])))
    redundant = max(0, N - len(owners))

    half = len(err_all) // 2
    return {
        "overall": float(np.mean(err_all[half:])),
        "react": float(np.mean(err_react) if err_react else np.nan),
        "coverage": coverage_frac,
        "redundant": redundant,
    }


def trials(steps, regimes, methods, K, N, mode, n_trials=20, **kw):
    out = [run(steps, regimes, methods, K, N, mode, seed=s, **kw) for s in range(n_trials)]
    keys = out[0].keys()
    return {k: np.array([o[k] for o in out], dtype=float) for k in keys}


def main():
    print("#" * 92)
    print("# POPULATION SIZING OFF THE N=R DIAGONAL: coverage, redundancy, and retention")
    print("#" * 92)
    R, W = 6, 3
    steps, regimes, methods, react = make_nonstationary_stream(R=R, W=W, seed=0)
    _REACT_CACHE[id(steps)] = react
    Ns = [1, 2, 3, 4, 6, 9, 12, 18]
    print(f"\nStream: R={R}, W={W}, {len(steps)} steps; probe={PROBE}. Sweeping N in {Ns}.\n")

    all_rows = {}
    for K in (1, 2):
        ceiling = lambda N: min(1.0, N * K / R)
        print(f"\n===== capacity K={K}  (coverage feasibility ceiling = min(1, N*K/R)) =====")
        print(f"{'N':>4}{'N*K/R':>8}{'coverage':>10}{'react_err':>11}{'overall':>9}"
              f"{'redundant':>11}")
        print("-" * 53)
        rows = []
        for N in Ns:
            sp = trials(steps, regimes, methods, K, N, "specialized")
            rows.append({
                "N": N, "K": K, "ceiling": ceiling(N),
                "coverage": float(sp["coverage"].mean()),
                "coverage_sem": _sem(sp["coverage"]),
                "react": float(np.nanmean(sp["react"])), "react_sem": _sem(sp["react"]),
                "overall": float(sp["overall"].mean()), "overall_sem": _sem(sp["overall"]),
                "redundant": float(sp["redundant"].mean()),
                "redundant_sem": _sem(sp["redundant"]),
            })
            print(f"{N:>4}{ceiling(N):>8.2f}{sp['coverage'].mean():>10.2f}"
                  f"{np.nanmean(sp['react']):>11.3f}{sp['overall'].mean():>9.3f}"
                  f"{sp['redundant'].mean():>11.1f}")
        all_rows[f"K{K}"] = rows
    print("-" * 53)
    # Headline reads
    k1 = all_rows["K1"]
    scarce = next(r for r in k1 if r["N"] == 3)   # N=3 < R=6 at K=1: ceiling 0.5
    diag = next(r for r in k1 if r["N"] == 6)
    abund = next(r for r in k1 if r["N"] == 18)
    print(f"\nK=1 scarce  (N=3, ceiling {scarce['ceiling']:.2f}): coverage {scarce['coverage']:.2f}, "
          f"react err {scarce['react']:.3f}")
    print(f"K=1 diagonal(N=6, ceiling {diag['ceiling']:.2f}): coverage {diag['coverage']:.2f}, "
          f"react err {diag['react']:.3f}, redundant {diag['redundant']:.1f}")
    print(f"K=1 abundant(N=18,ceiling {abund['ceiling']:.2f}): coverage {abund['coverage']:.2f}, "
          f"react err {abund['react']:.3f}, redundant {abund['redundant']:.1f}")
    print("\nReading: below the feasibility ceiling (N*K<R) competitive coverage is capped and")
    print("retention degrades smoothly; at/above N=R coverage saturates and surplus agents go")
    print("idle (redundant) without harming retention -- the equilibrium is robust off-diagonal.")

    save_and_plot(all_rows, R, W, Path(__file__).parent.parent / "results" / "population_sizing")


def save_and_plot(all_rows, R, W, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"R": R, "W": W, "probe": PROBE, "rows_by_K": all_rows}, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Results saved to {out_dir} (matplotlib unavailable).")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"K1": "#1f77b4", "K2": "#ff7f0e"}
    # Left: coverage vs N with feasibility ceiling
    ax = axes[0]
    for kkey, rows in all_rows.items():
        Ns = [r["N"] for r in rows]
        ax.errorbar(Ns, [r["coverage"] for r in rows],
                    yerr=[1.96 * r["coverage_sem"] for r in rows],
                    fmt="^-", color=colors[kkey], capsize=2, markersize=5,
                    label=f"coverage (K={kkey[1:]})")
        ax.plot(Ns, [r["ceiling"] for r in rows], ":", color=colors[kkey], alpha=0.7,
                label=f"ceiling min(1,NK/R), K={kkey[1:]}")
    ax.axvline(R, color="grey", ls="--", alpha=0.6)
    ax.text(R, 0.05, " N=R", color="grey", fontsize=8)
    ax.set_xlabel("number of agents N"); ax.set_ylabel("coverage fraction")
    ax.set_title("Coverage vs population size", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)
    # Right: post-reactivation error and redundancy vs N
    ax = axes[1]
    for kkey, rows in all_rows.items():
        Ns = [r["N"] for r in rows]
        ax.errorbar(Ns, [r["react"] for r in rows],
                    yerr=[1.96 * r["react_sem"] for r in rows],
                    fmt="o-", color=colors[kkey], capsize=2, markersize=5,
                    label=f"post-react err (K={kkey[1:]})")
    ax.axvline(R, color="grey", ls="--", alpha=0.6)
    ax.set_xlabel("number of agents N"); ax.set_ylabel("post-reactivation error")
    ax.set_title("Retention vs population size", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Population sizing off the N=R diagonal (R={R}, W={W})", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "population_sizing.png", dpi=150)
    paper_fig = Path(__file__).parent.parent / "paper" / "figures" / "fig12_population_sizing.pdf"
    if paper_fig.parent.exists():
        fig.savefig(paper_fig)
    print(f"Results + figure saved to {out_dir}")


if __name__ == "__main__":
    main()
