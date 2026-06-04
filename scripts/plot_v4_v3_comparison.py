"""
Plot the V4 vs V3 comparison results from ``exp_v4_v3_comparison.py``.

Produces three figures from the per-domain trajectories:
  1. SI trajectory means with shaded standard deviation across seeds,
     V4 and V3 overlaid per domain.
  2. Final-SI bar chart with error bars across all domains, V4 and V3.
  3. V3 clamp-invocation count and pre-norm-sum drift per domain.

Usage
-----
    python scripts/plot_v4_v3_comparison.py \\
        --in-dir results/v4_v3_comparison \\
        --label "matched_eta_0p1"
    python scripts/plot_v4_v3_comparison.py \\
        --in-dir results/v4_v3_comparison_matched_rate \\
        --label "matched_first_order_rate"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(in_dir: Path) -> Dict:
    return json.loads((in_dir / "summary.json").read_text())


def load_trajectory(in_dir: Path, domain: str) -> Dict[str, List[List[float]]]:
    return json.loads((in_dir / "trajectories" / f"{domain}.json").read_text())


def plot_trajectories(in_dir: Path, summary: Dict, out_path: Path) -> None:
    domains = list(summary["results"].keys())
    n = len(domains)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows), sharey=True)
    axes = np.atleast_2d(axes).flatten()

    idx = -1
    for idx, domain in enumerate(domains):
        traj = load_trajectory(in_dir, domain)
        v4 = np.array(traj["v4_trajectories"])
        v3 = np.array(traj["v3_trajectories"])

        v4_mean = v4.mean(axis=0)
        v4_std = v4.std(axis=0)
        v3_mean = v3.mean(axis=0)
        v3_std = v3.std(axis=0)
        x = np.linspace(1, v4.shape[1], v4.shape[1])

        ax = axes[idx]
        ax.plot(x, v4_mean, color="tab:blue", lw=2, label="V4 (EG)")
        ax.fill_between(x, v4_mean - v4_std, v4_mean + v4_std, color="tab:blue", alpha=0.18)
        ax.plot(x, v3_mean, color="tab:red", lw=2, label="V3 (legacy)")
        ax.fill_between(x, v3_mean - v3_std, v3_mean + v3_std, color="tab:red", alpha=0.18)
        ax.set_title(domain)
        ax.set_xlabel("Checkpoint")
        ax.grid(True, alpha=0.3)
        if idx % cols == 0:
            ax.set_ylabel("Population mean SI")
        ax.legend(loc="best", fontsize=8)

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    cfg = summary["config"]
    fig.suptitle(
        f"V4 (EG) vs V3 (legacy) SI trajectories  -- eta = {cfg['lr']},  "
        f"N = {cfg['n_agents']},  seeds = {cfg['n_seeds']}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_final_si_bars(summary: Dict, out_path: Path) -> None:
    domains = list(summary["results"].keys())
    v4_means = [summary["results"][d]["v4_eg"]["mean_si"] for d in domains]
    v4_stds = [summary["results"][d]["v4_eg"]["std_si"] for d in domains]
    v3_means = [summary["results"][d]["v3_additive"]["mean_si"] for d in domains]
    v3_stds = [summary["results"][d]["v3_additive"]["std_si"] for d in domains]

    x = np.arange(len(domains))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width / 2, v4_means, width, yerr=v4_stds, color="tab:blue", alpha=0.85, label="V4 (EG)")
    ax.bar(x + width / 2, v3_means, width, yerr=v3_stds, color="tab:red", alpha=0.85, label="V3 (legacy)")
    ax.set_xticks(x)
    ax.set_xticklabels(domains, rotation=15)
    ax.set_ylabel("Final population mean SI")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    cfg = summary["config"]
    ax.set_title(
        f"Final SI by domain  -- eta = {cfg['lr']},  "
        f"seeds = {cfg['n_seeds']},  iterations = {cfg['n_iterations']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_v3_diagnostics(summary: Dict, out_path: Path) -> None:
    domains = list(summary["results"].keys())
    clamps = [summary["results"][d]["v3_additive"]["total_clamp_invocations"] for d in domains]
    premass = [summary["results"][d]["v3_additive"]["mean_premass_sum"] for d in domains]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(domains))

    ax1.bar(x, clamps, color="tab:red", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains, rotation=15)
    ax1.set_ylabel("V3 clamp invocations (total across seeds)")
    ax1.set_title("V3 max(0.01, .) clamp -- fires thousands of times under matched config")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, premass, color="tab:red", alpha=0.85)
    ax2.axhline(1.0, color="tab:blue", lw=2, linestyle="--", label="V4 (EG): pre-norm sum drift bounded above by exp(eta), strict lower bound 1")
    ax2.set_xticks(x)
    ax2.set_xticklabels(domains, rotation=15)
    ax2.set_ylabel("V3 mean pre-norm sum on winning rounds")
    ax2.set_title("V3 mass drift -- pre-norm sum < 1 each winning round (Prop. 9.1)")
    ax2.set_ylim(0.85, 1.05)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--label", type=str, default="comparison",
                   help="Label used in output filenames")
    args = p.parse_args()

    summary = load_summary(args.in_dir)
    out_root = args.in_dir / "plots"
    out_root.mkdir(parents=True, exist_ok=True)

    plot_trajectories(args.in_dir, summary,
                      out_root / f"trajectories_{args.label}.png")
    plot_final_si_bars(summary,
                       out_root / f"final_si_bars_{args.label}.png")
    plot_v3_diagnostics(summary,
                        out_root / f"v3_diagnostics_{args.label}.png")

    print(f"Wrote plots to {out_root}")
    for f in sorted(out_root.glob(f"*{args.label}*.png")):
        print(f"  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
