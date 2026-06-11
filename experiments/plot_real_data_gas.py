#!/usr/bin/env python3
"""Render the real-data (UCI Gas Sensor Array Drift) result figure for the papers.

Reads gas_real.json + robustness_sweep_real.json (produced by
exp_real_data_gas.py --data ... and exp_robustness_sweep.py --data ...) and writes
paper/figures/fig_real_data_gas.{pdf,png}.

Panel A: post-reactivation accuracy by arm on the REAL stream -- the label-free
arms (GAUSE included) collapse to 0, the supervised full-capacity linear classifier
does NOT forget, and EWC's anchor hurts under drift.
Panel B: the EWC lambda sweep on real data -- anchoring monotonically degrades
retention and eventually diverges, the opposite of the benign-surrogate flat line.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
FIGDIR = HERE.parent / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

gas = json.load(open(HERE / "gas_real.json"))["rows"]
sweep = json.load(open(HERE / "robustness_sweep_real.json"))

order = ["gause_lf", "monolith_lf", "learned_div_lf",
         "oracle_fixed", "cl_naive", "cl_ewc", "cl_replay"]
labels = ["GAUSE-LF\n(ours)", "monolith-LF", "learned-div-LF",
          "oracle\n(labels)", "CL naive", "CL EWC", "CL replay"]
colors = {"label-free (ours)": "#1f77b4", "label-free": "#7fb3d5",
          "skyline (labels)": "#7f7f7f", "supervised CL": "#d62728"}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.2))

vals = [gas[k]["post_react_acc"] for k in order]
cols = [colors[gas[k]["kind"]] for k in order]
bars = axA.bar(range(len(order)), vals, color=cols, edgecolor="black", linewidth=0.6)
for i, v in enumerate(vals):
    axA.text(i, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
axA.set_xticks(range(len(order)))
axA.set_xticklabels(labels, fontsize=8)
axA.set_ylabel("post-reactivation accuracy")
axA.set_ylim(0, 0.8)
axA.set_title("(A) Real gas stream: who retains a dormant class?", fontsize=10)
axA.axhline(0, color="black", lw=0.8)
from matplotlib.patches import Patch
axA.legend(handles=[Patch(color="#1f77b4", label="label-free GAUSE"),
                    Patch(color="#7fb3d5", label="label-free baselines"),
                    Patch(color="#7f7f7f", label="oracle skyline"),
                    Patch(color="#d62728", label="supervised CL")],
           fontsize=7.5, loc="upper left")
axA.grid(axis="y", alpha=0.3)

ew = sweep["ewc"]["rows"]
xs = list(range(len(ew)))
ys = [r[1] for r in ew]
axB.plot(xs, ys, "o-", color="#d62728", label="EWC (anchor knob $\\lambda$)")
# mark divergence at the last point(s): W -> NaN at lambda=300
axB.annotate("diverges\n(NaN weights)", xy=(xs[-1], ys[-1]), xytext=(xs[-1]-1.4, ys[-1]+0.16),
             fontsize=7.5, ha="center",
             arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
axB.axhline(sweep["replay"]["best"], color="#2ca02c", lw=2, ls="--",
            label=f"naive = replay (flat {sweep['replay']['best']:.2f})")
axB.axhline(sweep["gause"]["mean"], color="#1f77b4", lw=2,
            label=f"GAUSE-LF (flat {sweep['gause']['mean']:.2f})")
axB.set_xticks(xs)
axB.set_xticklabels([str(r[0]) for r in ew], fontsize=8)
axB.set_xlabel("EWC $\\lambda$")
axB.set_ylabel("post-reactivation accuracy")
axB.set_ylim(-0.03, 0.8)
axB.set_title("(B) On real drift, EWC's anchor HURTS", fontsize=10)
axB.legend(fontsize=7.5, loc="upper right")
axB.grid(alpha=0.3)

fig.suptitle("UCI Gas Sensor Array Drift (13,910 samples, 128-d, 6 classes, 10 batches / 36 months)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"fig_real_data_gas.{ext}", dpi=140)
print(f"Saved -> {FIGDIR/'fig_real_data_gas.pdf'}")
