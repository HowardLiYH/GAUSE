#!/usr/bin/env python3
"""
E5 --- Split-CIFAR-100 continual learning: the surviving claim on a STANDARD,
citable benchmark with real images and gradient-trained CNN experts.

Why this experiment. On real gas-sensor *features* a full-capacity linear classifier
does not catastrophically forget, so the GAUSE retention dissociation lives in the
*representation-sharing* regime --- where learned features interfere. The
permuted-digits experiment (exp_function_approx_cl.py) already shows the dissociation
there on real digit images. This experiment puts that single surviving claim on a
harder, standard continual-learning benchmark the CL community recognizes:
Split-CIFAR-100 with small convolutional experts trained by SGD.

Design (consistent with the digits experiment). Split-CIFAR-100 gives R disjoint
10-way tasks (task t = a fixed block of 10 CIFAR-100 classes, relabelled 0..9). The
GAUSE retention metric needs *recurrence* (dormant -> reactivate), which standard
single-pass Split-CIFAR lacks, so we stream the R tasks through the SAME sliding
active-window schedule (make_task_stream) used everywhere else: each task is active
for a window, goes dormant, and reactivates. We measure post-reactivation test error
(did the serving expert retain the task across dormancy?) and overall test error.

Arms (identical assignment logic to exp_function_approx_cl.py, CNN substituted for MLP):
  monolith : 1 CNN, fine-tuned on whatever task is active (reward-driven by necessity).
  moe      : E CNN experts + a learned eps-greedy gate (reward-driven assignment).
  gause    : E CNN experts coupled only by winner-take-all on batch accuracy +
             EG niche affinity (reward-INDEPENDENT converged assignment).

Prediction. monolith and moe forget dormant tasks (high post-react error, closing only
as E -> R); gause retains (flat, low) --- the same dissociation as digits/tabular,
now on Split-CIFAR-100. An honest null (no dissociation on CIFAR) would BOUND the
neural claim to easy benchmarks; we report whatever we get.

Self-contained except for: torch, torchvision (CIFAR-100 download), sklearn(=scipy.stats).
Runs on Apple MPS / CUDA / CPU. Reuses make_task_stream, eg_eta, _sem from the digits
module so the two neural experiments share one protocol.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from exp_function_approx_cl import make_task_stream, eg_eta, _sem  # shared protocol

import torchvision
import torchvision.transforms as T

DATA = Path(__file__).resolve().parent.parent / "data" / "cifar"
BATCH = 64
TEST_BATCH = 256
N_STEPS = 15          # SGD steps per task-visit (CIFAR is harder than digits)
SGD_LR = 0.05


def get_device(pref=None):
    if pref:
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------------- data --------------------------------------- #
def load_split_cifar(R, classes_per_task, device, seed=0):
    """R disjoint tasks from CIFAR-100; task t holds `classes_per_task` classes,
    relabelled 0..C-1. Returns list of (Xtr,ytr,Xte,yte) float tensors on `device`."""
    tf = T.Compose([T.ToTensor(),
                    T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))])
    tr = torchvision.datasets.CIFAR100(str(DATA), train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR100(str(DATA), train=False, download=True, transform=tf)

    def to_tensor(ds):
        X = torch.stack([ds[i][0] for i in range(len(ds))])      # [N,3,32,32]
        y = torch.tensor([ds[i][1] for i in range(len(ds))])
        return X, y

    Xtr_all, ytr_all = to_tensor(tr)
    Xte_all, yte_all = to_tensor(te)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(100)
    tasks = []
    for t in range(R):
        cls = perm[t * classes_per_task:(t + 1) * classes_per_task]
        remap = {int(c): j for j, c in enumerate(cls)}
        mtr = np.isin(ytr_all.numpy(), cls)
        mte = np.isin(yte_all.numpy(), cls)
        ytr = torch.tensor([remap[int(c)] for c in ytr_all.numpy()[mtr]])
        yte = torch.tensor([remap[int(c)] for c in yte_all.numpy()[mte]])
        tasks.append((Xtr_all[mtr].to(device), ytr.to(device),
                      Xte_all[mte].to(device), yte.to(device)))
    return tasks


# ------------------------------- model -------------------------------------- #
class SmallCNN(nn.Module):
    def __init__(self, n_out=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32x16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64x8x8
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128),
                                  nn.ReLU(), nn.Linear(128, n_out))

    def forward(self, x):
        return self.head(self.features(x))


def _make_expert(seed, n_out, device):
    torch.manual_seed(seed)
    m = SmallCNN(n_out).to(device)
    opt = torch.optim.SGD(m.parameters(), lr=SGD_LR, momentum=0.9)
    return m, opt


def _train_block(model, opt, Xtr, ytr, rng, n_steps=N_STEPS):
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    n = Xtr.shape[0]
    for _ in range(n_steps):
        idx = torch.from_numpy(rng.choice(n, size=min(BATCH, n), replace=False)).to(Xtr.device)
        opt.zero_grad()
        loss = loss_fn(model(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()


@torch.no_grad()
def _acc(model, X, y, rng, k):
    model.eval()
    n = X.shape[0]
    idx = torch.from_numpy(rng.choice(n, size=min(k, n), replace=False)).to(X.device)
    return float((model(X[idx]).argmax(1) == y[idx]).float().mean())


# ------------------------------- runner ------------------------------------- #
def run(tasks, stream, E, mode, device, lam=0.5, gate_lr=0.3, eps_route=0.15, seed=0):
    R = len(tasks)
    rng = np.random.default_rng(seed)
    N = 1 if mode == "monolith" else E
    n_out = int(max(int(t[1].max()) for t in tasks)) + 1
    experts = [_make_expert(seed * 131 + i, n_out, device) for i in range(N)]
    gate = np.zeros((R, N))
    affinity = np.full((N, R), 1.0 / R) + 0.05 * rng.random((N, R))
    affinity /= affinity.sum(1, keepdims=True)

    err_overall, err_react = [], []
    half = len(stream) // 2
    for step, (epoch, t, is_react) in enumerate(stream):
        Xtr, ytr, Xte, yte = tasks[t]
        if mode == "monolith":
            w = 0
        elif mode == "moe":
            w = int(rng.integers(N)) if rng.random() < eps_route else int(np.argmax(gate[t]))
        else:  # gause winner-take-all on niche-shaped batch accuracy
            q = np.array([_acc(experts[i][0], Xtr, ytr, rng, BATCH) for i in range(N)])
            commit = affinity.max(1) - affinity[:, t]
            w = int(np.argmax(q + lam * affinity[:, t] - lam * commit))

        model, opt = experts[w]
        _train_block(model, opt, Xtr, ytr, rng)
        q_w = _acc(model, Xtr, ytr, rng, BATCH)

        if mode == "moe":
            gate[t, w] += gate_lr * (q_w - 0.1)
        elif mode == "gause":
            affinity[w, t] *= np.exp(eg_eta(R) * q_w)
            affinity[w] /= affinity[w].sum()

        if mode == "monolith":
            served = 0
        elif mode == "moe":
            served = int(np.argmax(gate[t]))
        else:
            served = int(np.argmax(affinity[:, t]))
        err = 1.0 - _acc(experts[served][0], Xte, yte, rng, TEST_BATCH)
        if step >= half:
            err_overall.append(err)
        if is_react:
            err_react.append(err)
    return (float(np.mean(err_overall)) if err_overall else np.nan,
            float(np.mean(err_react)) if err_react else np.nan)


def trials(tasks, stream, E, mode, device, n_trials, **kw):
    out = [run(tasks, stream, E, mode, device, seed=s, **kw) for s in range(n_trials)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=5, help="number of tasks (recurring regimes)")
    ap.add_argument("--classes-per-task", type=int, default=10)
    ap.add_argument("--W", type=int, default=3)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = get_device(args.device)
    torch.set_num_threads(4)

    print("#" * 90)
    print("# E5 --- SPLIT-CIFAR-100 CONTINUAL LEARNING (gradient-trained CNN experts)")
    print(f"# device={device}  R={args.R} tasks x {args.classes_per_task} classes, "
          f"W={args.W}, trials={args.trials}")
    print("#" * 90)
    tasks = load_split_cifar(args.R, args.classes_per_task, device, seed=0)
    stream = make_task_stream(R=args.R, W=args.W, seed=0)
    n_react = sum(1 for _, _, ir in stream if ir)
    print(f"\nSplit-CIFAR-100: {len(stream)} task-visits, {n_react} reactivations; "
          f"SmallCNN (2 conv + 2 fc), SGD lr={SGD_LR}.\n")
    print(f"{'E':>3}{'monolith':>16}{'MoE router':>16}{'GAUSE':>16}")
    print(f"{'(experts)':>9}{'over/react':>16}{'over/react':>16}{'over/react':>16}")
    print("-" * 60)

    # monolith ignores E (N=1) -> compute once and reuse
    mo, mr = trials(tasks, stream, 1, "monolith", device, args.trials)
    rows = []
    for E in range(1, args.R + 1):
        go, gr = trials(tasks, stream, E, "moe", device, args.trials)
        so, sr = trials(tasks, stream, E, "gause", device, args.trials)
        _t, p_react = stats.ttest_ind(sr, gr, alternative="less")
        rows.append({"E": E,
                     "monolith_overall": float(mo.mean()), "monolith_react": float(mr.mean()),
                     "moe_overall": float(go.mean()), "moe_react": float(gr.mean()),
                     "gause_overall": float(so.mean()), "gause_react": float(sr.mean()),
                     "gause_vs_moe_react_p": float(p_react),
                     "monolith_overall_sem": _sem(mo), "monolith_react_sem": _sem(mr),
                     "moe_overall_sem": _sem(go), "moe_react_sem": _sem(gr),
                     "gause_overall_sem": _sem(so), "gause_react_sem": _sem(sr)})
        print(f"{E:>3}      {f'{mo.mean():.3f}/{mr.mean():.3f}':>16}"
              f"{f'{go.mean():.3f}/{gr.mean():.3f}':>16}"
              f"{f'{so.mean():.3f}/{sr.mean():.3f}':>16}", flush=True)
    print("-" * 60)
    r = rows[-1]
    red = (r["moe_react"] - r["gause_react"]) / r["moe_react"] * 100 if r["moe_react"] else float("nan")
    print(f"E={args.R} (=R): post-react test error  GAUSE={r['gause_react']:.3f}  "
          f"MoE router={r['moe_react']:.3f} ({red:+.1f}% lower for GAUSE, "
          f"p={r['gause_vs_moe_react_p']:.2g})")

    out_dir = Path(__file__).parent.parent / "results" / "split_cifar_cl"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump({"benchmark": "Split-CIFAR-100", "R": args.R,
               "classes_per_task": args.classes_per_task, "W": args.W,
               "device": str(device), "trials": args.trials, "rows": rows},
              open(out_dir / "results.json", "w"), indent=2)
    print(f"\nSaved -> {out_dir/'results.json'}")
    _plot(rows, args.R, args.W, out_dir)


def _plot(rows, R, W, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    es = [r["E"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (suf, title) in zip(axes, [("overall", "Overall test error"),
                                       ("react", "Post-reactivation test error")]):
        for key, style, lab, col in [("monolith", "o-", "monolith (1 CNN)", "#d62728"),
                                     ("moe", "v-.", "MoE router", "#9467bd"),
                                     ("gause", "^-", "GAUSE (ours)", "#1f77b4")]:
            ax.errorbar(es, [r[f"{key}_{suf}"] for r in rows],
                        yerr=[1.96 * r[f"{key}_{suf}_sem"] for r in rows],
                        fmt=style, label=lab, color=col, capsize=2, markersize=5)
        ax.set_title(title, fontsize=10); ax.set_xlabel("number of experts E (capacity)")
        ax.set_ylabel("test error (1 - acc)"); ax.set_xticks(es); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Split-CIFAR-100 CL (R={R}, W={W}): router forgetting on a standard benchmark",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "split_cifar_cl.png", dpi=150)
    paper_fig = Path(__file__).parent.parent / "paper" / "figures" / "fig_split_cifar_cl.pdf"
    if paper_fig.parent.exists():
        fig.savefig(paper_fig)
    print(f"Saved figure -> {out_dir/'split_cifar_cl.png'} and paper/figures/fig_split_cifar_cl.pdf")


if __name__ == "__main__":
    main()
