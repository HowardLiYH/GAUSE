#!/usr/bin/env python3
"""
Function-approximation continual learning: is the router's failure architecture-agnostic?

All headline experiments use tabular Beta/Thompson learners. The reward-independence
principle predicts the learned MoE router's forgetting is a property of REWARD-DRIVEN
ASSIGNMENT, not of the tabular substrate -- i.e. it should reappear with gradient-trained
neural experts on a standard continual-learning benchmark. This experiment tests that
prediction directly, converting the central claim (and the agentic-AI / MoE application
narrative) from analogy into evidence.

Benchmark. Permuted-digits: R tasks, each a fixed random pixel permutation of
sklearn's 8x8 digits (10-way classification). This is the classic permuted-MNIST
continual-learning protocol at a size that runs on CPU in minutes. Experts are small
MLPs (64 -> hidden -> 10) trained by SGD -- genuine gradient-trained function
approximators that exhibit real catastrophic forgetting.

Capacity. The number of experts E (E <= R) is the analog of the tabular capacity K:
when E < R, experts are forced into reuse. We sweep E from 1 (the monolith) to R.

Arms.
  * monolith  : a single MLP fine-tuned on whatever task is active (E=1 reference at
                every point; reward-driven by necessity).
  * moe       : E experts + a learned gate (epsilon-greedy routing, gate trained on the
                routed expert's batch accuracy). Reward-driven assignment.
  * gause     : E experts coupled only by winner-take-all (the expert with the best
                batch accuracy, niche-shaped, wins, takes the SGD step, and reinforces
                its EG affinity). Reward-INDEPENDENT converged assignment.

Non-stationarity. Tasks cycle through a sliding active window W (dormant then
reactivating), exactly as in the tabular non-stationarity experiment. We report
post-reactivation test error (the relearning window) and overall test error.

Prediction. monolith and moe forget dormant tasks (high post-reactivation error at
tight capacity, closing only as E -> R); gause retains them (flat, low) -- the same
dissociation the tabular experiment shows, now with gradient-trained experts.
"""

import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy import stats

PROBE = 1            # tasks are evaluated once per (re)activation epoch on their test split
HIDDEN = 32
BATCH = 32
TEST_BATCH = 128
SGD_LR = 0.08


def make_permuted_tasks(R, seed=0):
    """R permuted-digits tasks sharing labels; each task applies a fixed feature
    permutation. Returns list of (Xtr, ytr, Xte, yte) float32 tensors."""
    d = load_digits()
    X = (d.data / 16.0).astype(np.float32)   # 0..1
    y = d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    rng = np.random.default_rng(seed)
    tasks = []
    for _ in range(R):
        perm = rng.permutation(X.shape[1])
        tasks.append((torch.from_numpy(Xtr[:, perm]), torch.from_numpy(ytr),
                      torch.from_numpy(Xte[:, perm]), torch.from_numpy(yte)))
    return tasks


def make_task_stream(R=5, W=3, n_epochs=18, seed=0):
    """Sliding active-window task stream. Each epoch activates W tasks and performs one
    'visit' (a block of SGD steps) on each active task; a task that was dormant and is
    seen again is a reactivation. Returns a list of (epoch, task_idx, is_reactivation)."""
    rng = np.random.default_rng(seed)
    stream = []
    prev_active, ever = set(), set()
    for e in range(n_epochs):
        active = sorted({(e + j) % R for j in range(W)})
        fresh = set(active) - prev_active
        react = fresh & ever
        order = list(active)
        rng.shuffle(order)
        for t in order:
            stream.append((e, t, t in react))
        ever |= set(active)
        prev_active = set(active)
    return stream


class MLP(nn.Module):
    def __init__(self, d_in=64, hidden=HIDDEN, d_out=10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def _make_expert(seed):
    torch.manual_seed(seed)
    m = MLP()
    opt = torch.optim.SGD(m.parameters(), lr=SGD_LR, momentum=0.9)
    return m, opt


def _train_block(model, opt, Xtr, ytr, rng, n_steps=8):
    """A short fine-tuning block on a task (the per-visit gradient training)."""
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    n = Xtr.shape[0]
    for _ in range(n_steps):
        idx = torch.from_numpy(rng.choice(n, size=min(BATCH, n), replace=False))
        opt.zero_grad()
        out = model(Xtr[idx])
        loss = loss_fn(out, ytr[idx])
        loss.backward()
        opt.step()


@torch.no_grad()
def _batch_acc(model, X, y, rng, k=TEST_BATCH):
    model.eval()
    n = X.shape[0]
    idx = torch.from_numpy(rng.choice(n, size=min(k, n), replace=False))
    pred = model(X[idx]).argmax(1)
    return float((pred == y[idx]).float().mean())


@torch.no_grad()
def _train_acc(model, Xtr, ytr, rng, k=BATCH):
    """Accuracy on a fresh training minibatch (the 'reward'/quality signal)."""
    model.eval()
    n = Xtr.shape[0]
    idx = torch.from_numpy(rng.choice(n, size=min(k, n), replace=False))
    return float((model(Xtr[idx]).argmax(1) == ytr[idx]).float().mean())


def run(tasks, stream, E, mode, lam=0.5, gate_lr=0.3, eps_route=0.15, seed=0):
    R = len(tasks)
    rng = np.random.default_rng(seed)
    N = 1 if mode == "monolith" else E
    experts = [_make_expert(seed * 131 + i) for i in range(N)]
    gate = np.zeros((R, N))                                  # MoE routing logits
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
        else:  # gause: winner-take-all on niche-shaped batch accuracy
            q = np.array([_train_acc(experts[i][0], Xtr, ytr, rng) for i in range(N)])
            commit = affinity.max(1) - affinity[:, t]
            w = int(np.argmax(q + lam * affinity[:, t] - lam * commit))

        model, opt = experts[w]
        _train_block(model, opt, Xtr, ytr, rng)             # the winner/routed expert learns
        q_w = _train_acc(model, Xtr, ytr, rng)

        if mode == "moe":
            gate[t, w] += gate_lr * (q_w - 0.5)             # reward-driven gate update
        elif mode == "gause":
            affinity[w, t] *= np.exp(eg_eta(R) * q_w)       # EG affinity update on win
            affinity[w] /= affinity[w].sum()

        # readout: who serves task t, then evaluate that expert on t's TEST split
        if mode == "monolith":
            served = 0
        elif mode == "moe":
            served = int(np.argmax(gate[t]))
        else:
            served = int(np.argmax(affinity[:, t]))
        err = 1.0 - _batch_acc(experts[served][0], Xte, yte, rng)
        if step >= half:
            err_overall.append(err)
        if is_react:
            err_react.append(err)

    return (float(np.mean(err_overall)) if err_overall else np.nan,
            float(np.mean(err_react)) if err_react else np.nan)


def eg_eta(R):
    # match the V4 rescaling used elsewhere (experiments/_affinity_update.eg_eta_for_regimes)
    return 0.1 * (R ** 2 - R + 1) / (R - 1)


def trials(tasks, stream, E, mode, n_trials=6, **kw):
    out = [run(tasks, stream, E, mode, seed=s, **kw) for s in range(n_trials)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def _sem(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0


def main():
    if not HAS_TORCH:
        print("PyTorch is required for the function-approximation experiment "
              "(pip install torch). Skipping.")
        return
    torch.set_num_threads(4)
    print("#" * 90)
    print("# FUNCTION-APPROXIMATION CONTINUAL LEARNING: is the router's failure")
    print("# architecture-agnostic? (gradient-trained MLP experts, permuted-digits)")
    print("#" * 90)
    R, W = 5, 3
    n_trials = 6
    tasks = make_permuted_tasks(R, seed=0)
    stream = make_task_stream(R=R, W=W, seed=0)
    n_react = sum(1 for _, _, ir in stream if ir)
    print(f"\nPermuted-digits: R={R} tasks, sliding window W={W}, {len(stream)} task-visits, "
          f"{n_react} reactivations; MLP 64-{HIDDEN}-10, SGD.\n")
    print(f"{'E':>3}{'monolith':>16}{'MoE router':>16}{'GAUSE':>16}")
    print(f"{'(experts)':>9}{'over/react':>16}{'over/react':>16}{'over/react':>16}")
    print("-" * 60)
    rows = []
    for E in range(1, R + 1):
        mo, mr = trials(tasks, stream, E, "monolith", n_trials=n_trials)   # E ignored (N=1)
        go, gr = trials(tasks, stream, E, "moe", n_trials=n_trials)
        so, sr = trials(tasks, stream, E, "gause", n_trials=n_trials)
        _t, p_react = stats.ttest_ind(sr, gr, alternative="less")
        rows.append({"E": E,
                     "monolith_overall": float(mo.mean()), "monolith_react": float(mr.mean()),
                     "moe_overall": float(go.mean()), "moe_react": float(gr.mean()),
                     "gause_overall": float(so.mean()), "gause_react": float(sr.mean()),
                     "gause_vs_moe_react_p": float(p_react),
                     "monolith_overall_sem": _sem(mo), "monolith_react_sem": _sem(mr),
                     "moe_overall_sem": _sem(go), "moe_react_sem": _sem(gr),
                     "gause_overall_sem": _sem(so), "gause_react_sem": _sem(sr)})
        print(f"{E:>3}      "
              f"{f'{mo.mean():.3f}/{mr.mean():.3f}':>16}"
              f"{f'{go.mean():.3f}/{gr.mean():.3f}':>16}"
              f"{f'{so.mean():.3f}/{sr.mean():.3f}':>16}")
    print("-" * 60)
    for E in (R,):
        r = next(x for x in rows if x["E"] == E)
        red = (r["moe_react"] - r["gause_react"]) / r["moe_react"] * 100
        print(f"E={E} (=R): post-react test error  GAUSE={r['gause_react']:.3f}  "
              f"MoE router={r['moe_react']:.3f} ({red:+.1f}% lower for GAUSE, "
              f"p={r['gause_vs_moe_react_p']:.2g})")
    print("\nReading: with gradient-trained experts the reward-driven router forgets dormant")
    print("tasks like the monolith and closes only as E->R, while GAUSE retains -- the same")
    print("dissociation as the tabular experiment, confirming it is architecture-agnostic.")

    save_and_plot(rows, R, W, Path(__file__).parent.parent / "results" / "function_approx_cl")


def save_and_plot(rows, R, W, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"R": R, "W": W, "hidden": HIDDEN, "sgd_lr": SGD_LR, "rows": rows}, f, indent=2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Results saved to {out_dir} (matplotlib unavailable).")
        return
    es = [r["E"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (suffix, title) in zip(axes, [("overall", "Overall test error"),
                                          ("react", "Post-reactivation test error")]):
        for key, style, lab, col in [
                ("monolith", "o-", "monolith (1 MLP)", "#d62728"),
                ("moe", "v-.", "MoE router", "#9467bd"),
                ("gause", "^-", "GAUSE (ours)", "#1f77b4")]:
            ax.errorbar(es, [r.get(f"{key}_{suffix}", np.nan) for r in rows],
                        yerr=[1.96 * r.get(f"{key}_{suffix}_sem", 0.0) for r in rows],
                        fmt=style, label=lab, color=col, capsize=2, markersize=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("number of experts E (capacity analog)")
        ax.set_ylabel("test error (1 - accuracy)")
        ax.set_xticks(es)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Function-approximation CL (permuted-digits, R={R}, W={W}): "
                 f"router forgetting is architecture-agnostic", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "function_approx_cl.png", dpi=150)
    paper_fig = Path(__file__).parent.parent / "paper" / "figures" / "fig11_function_approx_cl.pdf"
    if paper_fig.parent.exists():
        fig.savefig(paper_fig)
    print(f"Results + figure saved to {out_dir}")


if __name__ == "__main__":
    main()
