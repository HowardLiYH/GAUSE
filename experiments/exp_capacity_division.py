#!/usr/bin/env python3
"""
Does emergent specialization beat a monolith when individual CAPACITY is bounded?

The earlier ablations (exp_route_b_validation.py) showed that, with unlimited
individual capacity and stationary regimes, a single regime-conditioned agent
matches or beats the specialized population -- specialization was not an accuracy
lever. But that is precisely the regime where a generalist is feasible.

The realistic and interesting condition is BOUNDED PER-AGENT CAPACITY: each agent
can master only K of the R regimes (finite memory / model capacity / attention).
When K < R, no single agent can be a competent generalist. The question becomes:
can a POPULATION of capacity-K agents, through emergent specialization, cover the
regime space that no individual can -- and thereby beat a capacity-K monolith?

Design (regime-champion task: method m is uniquely best in regime m):
  * Each agent is competent only in its top-K niche-affinity regimes; elsewhere it
    has no expertise and must guess (uninformed method choice).
  * MONOLITH baseline: one capacity-K agent. It can cover at most K of R regimes.
  * SPECIALIZED population: N=R agents, capacity K each, competitive winner-take-all
    so niche affinity differentiates them across regimes (competitive exclusion).
  * Readout: each regime is served by the most-specialized agent; if none is
    competent there, the prediction is uninformed (the coverage penalty).

Prediction: at K=R both cover everything (tie). For K<R the monolith leaves R-K
regimes uncovered while the specialized population divides labor to cover (nearly)
all R -- a genuine, capacity-induced performance advantage for specialization.
This is the LeCun "intelligence is specialized" regime: under finite capacity,
division of labor is not optional.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.exp_route_b_coupled import make_synthetic_steps, cohens_d
from experiments.exp_domain_screen import build_steps, METHODS
from experiments._affinity_update import eg_eta_for_regimes


class CapacityAgent:
    def __init__(self, regimes: List[str], methods: List[str], capacity: int, rng,
                 fixed_competence=None):
        self.regimes = regimes
        self.methods = methods
        self.K = capacity
        self.rng = rng
        self.affinity = {r: 1.0 / len(regimes) + 1e-6 * rng.random() for r in regimes}
        self._renorm()
        self.beliefs = {r: {m: [1.0, 1.0] for m in methods} for r in regimes}
        # For the 'random' baseline: a fixed, randomly-assigned competence set
        # (diversity WITHOUT competition / coordination).
        self.fixed_competence = set(fixed_competence) if fixed_competence is not None else None

    def _renorm(self):
        tot = sum(self.affinity.values())
        for r in self.affinity:
            self.affinity[r] /= tot

    def competent_set(self):
        if self.fixed_competence is not None:
            return self.fixed_competence
        ranked = sorted(self.affinity, key=self.affinity.get, reverse=True)
        return set(ranked[:self.K])

    def select(self, regime: str, competent) -> str:
        if regime in competent:
            samples = {m: self.rng.beta(*self.beliefs[regime][m]) for m in self.methods}
            return max(samples, key=samples.get)
        return self.methods[self.rng.integers(len(self.methods))]  # uninformed guess

    def update_affinity(self, regime: str, reward: float, eta: float):
        self.affinity[regime] *= np.exp(eta * reward)
        self._renorm()

    def update_belief(self, regime: str, method: str, success: bool):
        self.beliefs[regime][method][0 if success else 1] += 1.0


def run_population(steps, regimes, methods, n_agents, capacity, mode, lam=0.3, seed=0):
    """mode in {'specialized', 'monolith', 'random'}.

      specialized : N competitive agents -> emergent division of labor (the algorithm).
      monolith    : a single capacity-K agent (can cover at most K of R regimes).
      random      : N agents with FIXED random capacity-K niches -- diversity WITHOUT
                    competition; the control for 'just more total capacity'.
    """
    rng = np.random.default_rng(seed)
    eta = eg_eta_for_regimes(len(regimes))
    inv_r = 1.0 / len(regimes)
    R = len(regimes)

    if mode == "monolith":
        agents = [CapacityAgent(regimes, methods, capacity, rng)]
    elif mode == "random":
        agents = [CapacityAgent(regimes, methods, capacity, rng,
                                fixed_competence=[regimes[i] for i in
                                                  rng.choice(R, size=capacity, replace=False)])
                  for _ in range(n_agents)]
    else:
        agents = [CapacityAgent(regimes, methods, capacity, rng) for _ in range(n_agents)]
    routed_err = []

    for regime, errs in steps:
        max_e, min_e = max(errs.values()), min(errs.values())
        span = (max_e - min_e) or 1.0
        comp = {id(a): a.competent_set() for a in agents}
        quality, chosen = {}, {}
        for a in agents:
            m = a.select(regime, comp[id(a)])
            chosen[id(a)] = m
            quality[id(a)] = (max_e - errs[m]) / span

        if mode == "monolith":
            winner = agents[0]
            winner.update_belief(regime, chosen[id(winner)], success=quality[id(winner)] >= 0.5)
            winner.update_affinity(regime, reward=quality[id(winner)], eta=eta)
        elif mode == "random":
            # No competition: every competent agent learns its assigned niche.
            for a in agents:
                if regime in comp[id(a)]:
                    a.update_belief(regime, chosen[id(a)], success=quality[id(a)] >= 0.5)
        else:
            scores = {id(a): quality[id(a)] + lam * (a.affinity[regime] - inv_r) for a in agents}
            winner = max(agents, key=lambda a: scores[id(a)])
            winner.update_belief(regime, chosen[id(winner)], success=quality[id(winner)] >= 0.5)
            winner.update_affinity(regime, reward=quality[id(winner)], eta=eta)

        # Readout: serve the regime with the most-specialized competent agent.
        competent_here = [a for a in agents if regime in comp[id(a)]]
        if competent_here:
            served = max(competent_here, key=lambda a: a.affinity[regime])
        else:
            served = max(agents, key=lambda a: a.affinity[regime])  # uncovered -> guesses
        routed_err.append(errs[served.select(regime, comp[id(served)])])

    half = len(routed_err) // 2
    return float(np.mean(routed_err[half:]))


def trials(steps, regimes, methods, n_agents, capacity, mode, n_trials=20):
    return np.array([run_population(steps, regimes, methods, n_agents, capacity,
                                    mode, seed=s) for s in range(n_trials)])


def sweep_synthetic(R=6, n_trials=20):
    print("=" * 84)
    print(f"SYNTHETIC regime-champion (R={R}, method m best in regime m), sweep capacity K")
    print("=" * 84)
    steps, regimes, methods = make_synthetic_steps(n_regimes=R, n_steps=2000,
                                                   separation=1.0, noise=0.3, seed=0)
    print(f"{'K':>3}{'monolith':>11}{'random-div':>12}{'specialized':>13}"
          f"{'spec vs mono%':>15}{'spec vs rand%':>15}{'p(vs rand)':>12}")
    print("-" * 95)
    rows = []
    for K in range(1, R + 1):
        mono = trials(steps, regimes, methods, R, K, "monolith", n_trials)
        rand = trials(steps, regimes, methods, R, K, "random", n_trials)
        spec = trials(steps, regimes, methods, R, K, "specialized", n_trials)
        g_mono = (mono.mean() - spec.mean()) / mono.mean() * 100
        g_rand = (rand.mean() - spec.mean()) / rand.mean() * 100
        _t, p_rand = stats.ttest_ind(spec, rand, alternative="less")
        rows.append({"K": K, "monolith": float(mono.mean()), "random": float(rand.mean()),
                     "specialized": float(spec.mean()), "spec_vs_mono_pct": float(g_mono),
                     "spec_vs_random_pct": float(g_rand), "p_vs_random": float(p_rand)})
        print(f"{K:>3}{mono.mean():>11.4f}{rand.mean():>12.4f}{spec.mean():>13.4f}"
              f"{g_mono:>15.2f}{g_rand:>15.2f}{p_rand:>12.2g}")
    print("-" * 95)
    print("'spec vs rand%' isolates the value of COMPETITION/coordination at equal total")
    print("capacity (N*K): specialized > random-diversity > monolith for K<R, tie at K=R.")
    return rows


def _load_solar():
    import pandas as pd
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "solar" / "openmeteo_real_irradiance.csv")
    df = df[df["location"] == "Denver_CO"].head(800)
    v = pd.to_numeric(df["ghi"], errors="coerce").to_numpy()
    hours = df["hour"].to_numpy()
    mask = ~np.isnan(v)
    v, hours = v[mask], hours[mask]

    def hr(h):
        h = int(h)
        return "night" if (h < 6 or h >= 20) else ("morning" if h < 9 else ("midday" if h < 16 else "evening"))
    return v, [hr(h) for h in hours]


def _load_traffic():
    import pandas as pd
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "traffic" / "nyc_taxi_real_hourly.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.head(800)
    v = df["trip_count"].to_numpy(float)

    def reg(ts):
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
    return v, [reg(t) for t in df["timestamp"]]


def real_domain(n_trials=20):
    print("\n" + "=" * 84)
    print("REAL domains: capacity sweep (clock-based regimes)")
    print("=" * 84)
    out = {}
    for name, (v, regimes_full) in [("traffic (NYC, R=6)", _load_traffic()),
                                    ("solar (Denver, R=3)", _load_solar())]:
        steps = build_steps(v, regimes_full)
        regimes = sorted(set(regimes_full[25:]))
        R = len(regimes)
        print(f"\n{name} -- regimes (R={R}): {regimes}")
        print(f"{'K':>3}{'monolith':>11}{'random-div':>12}{'specialized':>13}"
              f"{'spec vs mono%':>15}{'spec vs rand%':>15}{'p(vs rand)':>12}")
        print("-" * 95)
        rows = []
        for K in range(1, R + 1):
            mono = trials(steps, regimes, list(METHODS), R, K, "monolith", n_trials)
            rand = trials(steps, regimes, list(METHODS), R, K, "random", n_trials)
            spec = trials(steps, regimes, list(METHODS), R, K, "specialized", n_trials)
            g_mono = (mono.mean() - spec.mean()) / mono.mean() * 100
            g_rand = (rand.mean() - spec.mean()) / rand.mean() * 100
            _t, p_rand = stats.ttest_ind(spec, rand, alternative="less")
            rows.append({"K": K, "monolith": float(mono.mean()), "random": float(rand.mean()),
                         "specialized": float(spec.mean()), "spec_vs_mono_pct": float(g_mono),
                         "spec_vs_random_pct": float(g_rand), "p_vs_random": float(p_rand)})
            print(f"{K:>3}{mono.mean():>11.4f}{rand.mean():>12.4f}{spec.mean():>13.4f}"
                  f"{g_mono:>15.2f}{g_rand:>15.2f}{p_rand:>12.2g}")
        out[name] = {"R": R, "rows": rows}
    return out


def save_and_plot(synthetic_rows, real_results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"synthetic_regime_champion": {"R": 6, "rows": synthetic_rows},
               "real_domains": real_results}
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nResults saved to {out_dir} (matplotlib unavailable; skipped figure).")
        return

    panels = [("Synthetic (R=6, exclusive methods)", synthetic_rows)]
    panels += [(name, r["rows"]) for name, r in real_results.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, (title, rows) in zip(axes[0], panels):
        ks = [r["K"] for r in rows]
        ax.plot(ks, [r["monolith"] for r in rows], "o-", label="monolith (cap. K)", color="#d62728")
        ax.plot(ks, [r["random"] for r in rows], "s--", label="random diversity", color="#7f7f7f")
        ax.plot(ks, [r["specialized"] for r in rows], "^-", label="specialized (ours)", color="#1f77b4")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("per-agent capacity K")
        ax.set_ylabel("prediction error")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Bounded-capacity division of labor: specialization vs capacity-matched baselines",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / "capacity_sweep.png", dpi=150)
    paper_fig = Path(__file__).parent.parent / "paper" / "figures" / "fig6_capacity_division.pdf"
    if paper_fig.parent.exists():
        fig.savefig(paper_fig)
    print(f"\nResults + figure saved to {out_dir}")


def main():
    print("#" * 84)
    print("# CAPACITY-LIMITED DIVISION OF LABOR: does specialization beat a monolith")
    print("# when no single agent can master every regime?")
    print("#" * 84 + "\n")
    synthetic_rows = sweep_synthetic()
    real_results = real_domain()
    save_and_plot(synthetic_rows, real_results,
                  Path(__file__).parent.parent / "results" / "capacity_division")
    print("\nIf specialization wins for K<R, the strong claim holds under bounded capacity:")
    print("emergent division of labor achieves coverage a capacity-matched monolith cannot.")


if __name__ == "__main__":
    main()
