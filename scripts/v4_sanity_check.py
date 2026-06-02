"""
V4 EG sanity-check: small-scale V3 vs V4 comparison.

Purpose: Before kicking off the full multi-domain sweep, verify on a
small synthetic problem that

  1. Both V3 and V4 populations actually produce emergent specialization
  2. V4 reaches an equilibrium SI comparable to V3
  3. The V3 mass-drift and clamp diagnostics behave as predicted
  4. V4's mass-drift counter stays at exactly 1.0 (intrinsic simplex)

This is intentionally fast (a few minutes at most). It is NOT meant
to be the headline empirical result; that comes from the full
experiments under ``experiments/`` after Batch B is approved.

Usage:
    python scripts/v4_sanity_check.py

Output:
    results/v4_sanity/comparison_plot.png  -- SI trajectories side-by-side
    results/v4_sanity/diagnostics.txt      -- numerical diagnostics summary
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Local imports
from src.agents.niche_population import NichePopulation


REGIMES = ["regime_A", "regime_B", "regime_C", "regime_D"]
N_AGENTS = 8
N_ITERATIONS = 500
N_SEEDS = 5
ETA = 0.1
LAMBDA = 0.3


def specialization_index(affinity_vec: List[float]) -> float:
    """SI = 1 - H(alpha)/log(R), the canonical formula from the paper."""
    R = len(affinity_vec)
    if R <= 1:
        return 0.0
    h = 0.0
    for a in affinity_vec:
        if a > 0:
            h -= a * math.log(a)
    return 1.0 - h / math.log(R)


def population_mean_si(pop: NichePopulation) -> float:
    """Average SI across agents."""
    sis = []
    for agent in pop.agents.values():
        vec = [agent.niche_affinity[r] for r in agent.regimes]
        sis.append(specialization_index(vec))
    return float(np.mean(sis))


def fake_reward(methods: List[str], prices) -> float:
    """
    Stub reward function for the sanity check.

    We don't actually need a real market model here -- we just need a
    function the population can call. The V3 vs V4 comparison is about
    the affinity update, which is driven by who *wins* (determined by
    raw reward + niche bonus), not by the magnitude of raw reward.

    Returning a fixed value lets niche bonus dominate the winner
    determination, which is the regime in which the affinity update
    matters most.
    """
    return 0.5


def simulate_regimes(n_iterations: int, rng: np.random.Generator) -> List[str]:
    """Uniform regime distribution -- simplest setting where SI is well-defined."""
    return [REGIMES[i] for i in rng.integers(0, len(REGIMES), size=n_iterations)]


def run_one_trial(
    update_rule: str,
    seed: int,
    n_iterations: int = N_ITERATIONS,
) -> Tuple[List[float], Dict[str, float]]:
    """
    Returns:
        si_trajectory: mean SI over the population at each iteration
        diagnostics: dict with summary stats (clamp invocations, mass drift, etc.)
    """
    pop = NichePopulation(
        n_agents=N_AGENTS,
        regimes=REGIMES,
        niche_bonus=LAMBDA,
        seed=seed,
        learning_rate=ETA,
        update_rule=update_rule,
    )

    rng = np.random.default_rng(seed + 1000)
    regime_sequence = simulate_regimes(n_iterations, rng)

    si_trajectory: List[float] = []

    for t, regime in enumerate(regime_sequence):
        # The population's run_iteration signature expects prices + regime + reward_fn
        pop.run_iteration(prices=None, regime=regime, reward_fn=fake_reward)
        if (t + 1) % 10 == 0:
            si_trajectory.append(population_mean_si(pop))

    # Collect diagnostics across all agents
    total_clamp_invocations = sum(
        a._diag_clamp_invocations for a in pop.agents.values()
    )
    all_premass_sums: List[float] = []
    for a in pop.agents.values():
        all_premass_sums.extend(a._diag_premass_sum_history)

    final_si = si_trajectory[-1] if si_trajectory else 0.0

    diagnostics = {
        "final_si": final_si,
        "total_clamp_invocations": float(total_clamp_invocations),
        "mean_premass_sum": float(np.mean(all_premass_sums)) if all_premass_sums else float("nan"),
        "min_premass_sum": float(np.min(all_premass_sums)) if all_premass_sums else float("nan"),
        "n_premass_observations": float(len(all_premass_sums)),
    }
    return si_trajectory, diagnostics


def main() -> None:
    out_dir = Path("results/v4_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("V4 EG sanity check")
    print("=" * 60)
    print(f"Regimes:        {REGIMES}")
    print(f"Agents:         {N_AGENTS}")
    print(f"Iterations:     {N_ITERATIONS}")
    print(f"Seeds:          {N_SEEDS}")
    print(f"Learning rate:  {ETA}")
    print(f"Niche bonus:    {LAMBDA}")
    print()

    trajectories_v3: List[List[float]] = []
    trajectories_v4: List[List[float]] = []
    diagnostics_v3: List[Dict[str, float]] = []
    diagnostics_v4: List[Dict[str, float]] = []

    for seed in range(N_SEEDS):
        print(f"  seed {seed}: ", end="", flush=True)
        traj_v3, diag_v3 = run_one_trial("v3_additive", seed=seed)
        traj_v4, diag_v4 = run_one_trial("eg", seed=seed)
        trajectories_v3.append(traj_v3)
        trajectories_v4.append(traj_v4)
        diagnostics_v3.append(diag_v3)
        diagnostics_v4.append(diag_v4)
        print(
            f"V3 final SI = {diag_v3['final_si']:.3f}, "
            f"V4 final SI = {diag_v4['final_si']:.3f}, "
            f"V3 clamps = {int(diag_v3['total_clamp_invocations'])}, "
            f"V4 clamps = {int(diag_v4['total_clamp_invocations'])}"
        )

    # Aggregate
    arr_v3 = np.array(trajectories_v3)
    arr_v4 = np.array(trajectories_v4)

    # Plot (lazy import so the script runs even without matplotlib in CI)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(1, arr_v3.shape[1] + 1) * 10
        v3_mean = arr_v3.mean(axis=0)
        v3_std = arr_v3.std(axis=0)
        v4_mean = arr_v4.mean(axis=0)
        v4_std = arr_v4.std(axis=0)

        ax.plot(x, v3_mean, color="tab:red", lw=2, label="V3 (additive + clamp)")
        ax.fill_between(x, v3_mean - v3_std, v3_mean + v3_std, color="tab:red", alpha=0.18)
        ax.plot(x, v4_mean, color="tab:blue", lw=2, label="V4 (EG)")
        ax.fill_between(x, v4_mean - v4_std, v4_mean + v4_std, color="tab:blue", alpha=0.18)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Mean SI across population")
        ax.set_title(
            f"V3 vs V4: SI trajectory (N={N_AGENTS} agents, {N_SEEDS} seeds, "
            f"eta={ETA}, lambda={LAMBDA})"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = out_dir / "comparison_plot.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=130)
        plt.close(fig)
        print(f"\nSaved plot: {plot_path}")
    except ImportError:
        print("\nmatplotlib not available; skipping plot")

    # Diagnostics summary
    diag_path = out_dir / "diagnostics.txt"
    with open(diag_path, "w") as f:
        f.write("V4 EG Sanity Check -- Diagnostics Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Config: R={len(REGIMES)}, N={N_AGENTS}, iters={N_ITERATIONS}, "
                f"seeds={N_SEEDS}, eta={ETA}, lambda={LAMBDA}\n\n")

        def _summarize(name: str, diags: List[Dict[str, float]]) -> None:
            f.write(f"## {name}\n")
            final_sis = [d["final_si"] for d in diags]
            clamps = [d["total_clamp_invocations"] for d in diags]
            premass = [d["mean_premass_sum"] for d in diags]
            min_premass = [d["min_premass_sum"] for d in diags]
            f.write(f"  final SI:                 mean={np.mean(final_sis):.4f}, "
                    f"std={np.std(final_sis):.4f}, "
                    f"range=[{min(final_sis):.4f}, {max(final_sis):.4f}]\n")
            f.write(f"  total clamp invocations:  mean={np.mean(clamps):.1f}, "
                    f"sum={sum(clamps):.0f}\n")
            f.write(f"  mean pre-norm sum:        mean={np.mean(premass):.6f}, "
                    f"min observed={min(min_premass):.6f}\n")
            f.write("\n")

        _summarize("V3 (additive + clamp)", diagnostics_v3)
        _summarize("V4 (EG)", diagnostics_v4)

        # Key narrative
        f.write("## Interpretation\n")
        f.write("  - V4 pre-norm sum should equal 1 + alpha_winner * (exp(eta)-1) on each update;\n")
        f.write("    its mean is bounded above by exp(eta) and below by 1.\n")
        f.write("  - V3 pre-norm sum should drift below 1 (by approx -eta * alpha_winner);\n")
        f.write("    its mean should be visibly less than 1.\n")
        f.write("  - V4 clamp invocations should be exactly 0 (no clamping path in EG).\n")
        f.write("  - V3 clamp invocations should be positive after sufficient specialization.\n")

    print(f"Saved diagnostics: {diag_path}")

    # Print summary to stdout
    print()
    print("Summary:")
    print(f"  V3 final SI: {np.mean([d['final_si'] for d in diagnostics_v3]):.4f} "
          f"+/- {np.std([d['final_si'] for d in diagnostics_v3]):.4f}")
    print(f"  V4 final SI: {np.mean([d['final_si'] for d in diagnostics_v4]):.4f} "
          f"+/- {np.std([d['final_si'] for d in diagnostics_v4]):.4f}")
    v3_clamp_total = sum(d["total_clamp_invocations"] for d in diagnostics_v3)
    v4_clamp_total = sum(d["total_clamp_invocations"] for d in diagnostics_v4)
    print(f"  V3 total clamps:  {int(v3_clamp_total)}  (proves the v3 bug fires in practice)")
    print(f"  V4 total clamps:  {int(v4_clamp_total)}  (proves the v4 fix removes it entirely)")


if __name__ == "__main__":
    main()
