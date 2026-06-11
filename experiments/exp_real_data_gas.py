#!/usr/bin/env python3
"""
REAL-DATA validation of the GAUSE retention claim on the UCI Gas Sensor Array
Drift dataset --- a genuine recurring-regime, concept-drift stream with real
128-dim sensor features (16 sensors x 8 features), 6 gas classes, collected in
10 batches over 36 months.

Why this dataset. The paper's headline (distributed persistent memory under
non-stationarity) and its class-incremental claim were validated only on synthetic
streams where regime r is signalled 1:1 by method r. Here the regimes are real gas
classes, the non-stationarity is real sensor drift, and there is NO method<->regime
bijection: a label-free learner must infer the class from a drifting, overlapping
128-dim vector. This directly attacks the two stated weaknesses --- synthetic-data
reliance and the missing comparison to real continual-learning baselines.

Arms.
  Label-free (class-incremental) --- no class label at decision time:
    gause_lf        : N=R competitive prototype specialists (winner = nearest
                      prototype = implicit class estimate; only winner updates).
    monolith_lf     : 1 learner, K prototype slots (LRU) --- overwrites dormant
                      classes (forgets).
    learned_div_lf  : prototypes + an explicit identifiability/repulsion term.
    oracle_fixed    : agent i locked to class i via labels (SKYLINE).
  Supervised continual-learning baselines (label-given --- the honest CL competitors):
    cl_naive  : online multinomial logistic regression, plain SGD (forgets).
    cl_ewc    : + elastic weight consolidation (diagonal Fisher anchor at batch ends).
    cl_replay : + reservoir experience replay.

Metric. Post-reactivation accuracy: accuracy on a class in the probe window right
after it reappears from dormancy. Plus overall accuracy, and for the label-free
population, specialization/coverage of the recovered partition vs the hidden classes.

Usage.
  python exp_real_data_gas.py                 # faithful surrogate (no network needed)
  python exp_real_data_gas.py --data PATH     # PATH holds batch1.dat..batch10.dat
                                              # (UCI Gas Sensor Array Drift, free)
Self-contained: numpy only.
"""

import argparse, glob, json, os
from collections import OrderedDict, Counter, defaultdict
from pathlib import Path
import numpy as np

PROBE = 12
R = 6


# ----------------------------- data ----------------------------------------- #
def load_gas_real(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "batch*.dat")),
                   key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit())))
    if not files:
        raise FileNotFoundError(f"no batch*.dat in {data_dir}")
    X, y, b = [], [], []
    for bi, f in enumerate(files):
        for line in open(f):
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0])) - 1
            vec = np.zeros(128)
            for tok in parts[1:]:
                idx, val = tok.split(":")
                j = int(idx) - 1
                if 0 <= j < 128:
                    vec[j] = float(val)
            X.append(vec); y.append(cls); b.append(bi)
    X, y, b = np.asarray(X), np.asarray(y), np.asarray(b)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X, y, b


def make_surrogate(d=128, n_batches=10, per_class_batch=140, drift=0.75,
                   overlap=1.6, seed=0):
    """Offline stand-in: R Gaussian classes whose centroids random-walk across
    batches (sensor drift), with shared overlapping noise (no 1:1 feature<->class
    signature, so the label-free test is honest)."""
    rng = np.random.default_rng(seed)
    cent = rng.normal(0, 1.0, size=(R, d))
    X, y, b = [], [], []
    for bi in range(n_batches):
        cent = cent + rng.normal(0, drift, size=(R, d))
        for k in range(R):
            pts = cent[k] + rng.normal(0, overlap, size=(per_class_batch, d))
            X.append(pts); y.append(np.full(per_class_batch, k)); b.append(np.full(per_class_batch, bi))
    X, y, b = np.vstack(X), np.concatenate(y), np.concatenate(b)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return X, y, b


def build_stream(X, y, b, W=3, seed=0):
    rng = np.random.default_rng(seed)
    n_batches = int(b.max()) + 1
    idx_by = defaultdict(list)
    for i in range(len(y)):
        idx_by[(int(b[i]), int(y[i]))].append(i)
    for k in idx_by:
        rng.shuffle(idx_by[k])
    stream, prev_active, ever = [], set(), set()
    for e in range(n_batches):
        active = sorted({(e + j) % R for j in range(W)})
        fresh = (set(active) - prev_active) & ever
        pools = {k: list(idx_by.get((e, k), [])) for k in active}
        for k in active:
            if not pools[k]:
                allk = [i for (bb, kk), lst in idx_by.items() if kk == k for i in lst]
                rng.shuffle(allk); pools[k] = allk[:120]
        order, maxlen = [], max((len(v) for v in pools.values()), default=0)
        for s in range(maxlen):
            for k in active:
                if s < len(pools[k]):
                    order.append((pools[k][s], k))
        rng.shuffle(order)
        seen = Counter()
        for (i, k) in order:
            is_probe = (k in fresh) and (seen[k] < PROBE)
            seen[k] += 1
            stream.append((X[i], k, is_probe))
        prev_active, ever = set(active), ever | set(active)
    return stream


# ----------------------- label-free prototype arms -------------------------- #
class ProtoPop:
    def __init__(self, d, N, lr=0.15, mode="gause", div=0.4, rng=None):
        self.d, self.N, self.lr, self.mode, self.div = d, N, lr, mode, div
        self.rng = rng or np.random.default_rng(0)
        self.mu = [None] * N
        self.cnt = [Counter() for _ in range(N)]

    def _scores(self, x):
        return [(-np.sum((x - self.mu[a]) ** 2) if self.mu[a] is not None else -1e18)
                for a in range(self.N)]

    def observe_update(self, x, true_k, learn=True):
        for a in range(self.N):
            if self.mu[a] is None:
                self.mu[a] = x.copy(); self.cnt[a][true_k] += 1
                return
        s = self._scores(x); w = int(np.argmax(s))
        if learn:
            self.mu[w] = self.mu[w] + self.lr * (x - self.mu[w])
            if self.mode == "diversity":
                order = np.argsort(s)[::-1]
                if len(order) > 1:
                    r = int(order[1])
                    self.mu[r] = self.mu[r] - (self.div * 0.02) * (x - self.mu[r])
        self.cnt[w][true_k] += 1

    def predict(self, x):
        return int(np.argmax(self._scores(x)))

    def cls_of(self, a):
        return self.cnt[a].most_common(1)[0][0] if self.cnt[a] else -1

    def predict_class(self, x):
        return self.cls_of(self.predict(x))


class ProtoMonolith:
    def __init__(self, d, K, lr=0.15, rng=None):
        self.d, self.K, self.lr = d, K, lr
        self.protos = OrderedDict(); self.nxt = 0
        self.cnt = defaultdict(Counter)

    def observe_update(self, x, true_k, learn=True):
        # fill the K slots first (bounded memory); thereafter assign to nearest and
        # let drift + reuse overwrite whichever slot is nearest -- dormant classes,
        # never being nearest while active ones are, decay away (forgetting).
        if len(self.protos) < self.K:
            self.protos[self.nxt] = x.copy(); self.cnt[self.nxt][true_k] += 1; self.nxt += 1
            return
        slot = min(self.protos, key=lambda s: np.sum((x - self.protos[s]) ** 2))
        if learn:
            self.protos[slot] = self.protos[slot] + self.lr * (x - self.protos[slot])
            self.protos.move_to_end(slot)
        self.cnt[slot][true_k] += 1

    def predict_class(self, x):
        if not self.protos:
            return -1
        slot = min(self.protos, key=lambda s: np.sum((x - self.protos[s]) ** 2))
        return self.cnt[slot].most_common(1)[0][0] if self.cnt[slot] else -1


class OracleFixed:
    def __init__(self, d, lr=0.15):
        self.mu = {k: None for k in range(R)}; self.lr = lr

    def observe_update(self, x, true_k, learn=True):
        if self.mu[true_k] is None:
            self.mu[true_k] = x.copy()
        elif learn:
            self.mu[true_k] = self.mu[true_k] + self.lr * (x - self.mu[true_k])

    def predict_class(self, x):
        cand = [(k, m) for k, m in self.mu.items() if m is not None]
        return min(cand, key=lambda km: np.sum((x - km[1]) ** 2))[0] if cand else -1


# --------------- supervised continual-learning baselines -------------------- #
class OnlineSoftmax:
    def __init__(self, d, C=R, lr=0.05, mode="naive", ewc_lambda=20.0,
                 buf=400, replay_k=8, rng=None):
        self.d, self.C, self.lr, self.mode = d, C, lr, mode
        self.W = np.zeros((C, d)); self.bvec = np.zeros(C)
        self.ewc_lambda, self.buf_size, self.replay_k = ewc_lambda, buf, replay_k
        self.rng = rng or np.random.default_rng(0)
        self.star_W = None; self.fisher = None
        self.bX, self.by, self.seen = [], [], 0

    def _p(self, x):
        z = self.W @ x + self.bvec; z -= z.max(); e = np.exp(z); return e / e.sum()

    def _step(self, x, k, scale=1.0):
        p = self._p(x); p[k] -= 1.0
        gW = np.outer(p, x)
        if self.mode == "ewc" and self.star_W is not None:
            gW = gW + self.ewc_lambda * self.fisher * (self.W - self.star_W)
        self.W -= self.lr * scale * gW
        self.bvec -= self.lr * scale * p

    def observe_update(self, x, true_k, learn=True):
        if not learn:
            return
        self._step(x, true_k)
        if self.mode == "replay" and self.bX:
            for j in self.rng.integers(0, len(self.bX), size=min(self.replay_k, len(self.bX))):
                self._step(self.bX[j], self.by[j], scale=0.5)
            self.seen += 1
            if len(self.bX) < self.buf_size:
                self.bX.append(x.copy()); self.by.append(true_k)
            else:
                r = self.rng.integers(0, self.seen)
                if r < self.buf_size:
                    self.bX[r] = x.copy(); self.by[r] = true_k

    def consolidate(self, rX, ry):
        if self.mode != "ewc" or not rX:
            return
        F = np.zeros_like(self.W)
        for x, k in zip(rX, ry):
            p = self._p(x); p[k] -= 1.0; F += np.outer(p, x) ** 2
        F /= max(len(rX), 1)
        self.fisher = F if self.fisher is None else 0.5 * self.fisher + F
        self.star_W = self.W.copy()

    def predict_class(self, x):
        return int(np.argmax(self.W @ x + self.bvec))


# ------------------------------- runner ------------------------------------- #
def run_arm(name, stream, d, K=1, seed=0, **kw):
    rng = np.random.default_rng(seed)
    if name == "gause_lf":
        arm = ProtoPop(d, R, mode="gause", rng=rng)
    elif name == "monolith_lf":
        arm = ProtoMonolith(d, K=K, rng=rng)
    elif name == "learned_div_lf":
        arm = ProtoPop(d, R, mode="diversity", div=kw.get("div", 0.4), rng=rng)
    elif name == "oracle_fixed":
        arm = OracleFixed(d)
    elif name.startswith("cl_"):
        arm = OnlineSoftmax(d, mode=name.split("_", 1)[1], lr=kw.get("lr", 0.05),
                            ewc_lambda=kw.get("ewc_lambda", 20.0), buf=kw.get("buf", 400), rng=rng)
    else:
        raise ValueError(name)

    cp = np_ = ca = na = 0
    rX, ry = [], []
    for (x, k, probe) in stream:
        pred = arm.predict_class(x)            # test-then-train (online)
        ca += int(pred == k); na += 1
        if probe:
            cp += int(pred == k); np_ += 1
        arm.observe_update(x, k, learn=True)
        if isinstance(arm, OnlineSoftmax) and arm.mode == "ewc":
            rX.append(x); ry.append(k)
            if len(rX) > 600:
                arm.consolidate(rX, ry); rX, ry = [], []
    si = cov = np.nan
    if isinstance(arm, ProtoPop):
        pur = [arm.cnt[a].most_common(1)[0][1] / sum(arm.cnt[a].values())
               for a in range(arm.N) if arm.cnt[a]]
        si = float(np.mean(pur)) if pur else 0.0
        cov = len(set(arm.cls_of(a) for a in range(arm.N) if arm.cls_of(a) >= 0)) / R
    return cp / max(np_, 1), ca / max(na, 1), si, cov


def trials(name, stream, d, n=12, **kw):
    return np.array([run_arm(name, stream, d, seed=s, **kw) for s in range(n)])


def sem(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--trials", type=int, default=12)
    args = ap.parse_args()
    if args.data:
        X, y, b = load_gas_real(args.data); src = f"UCI gas-sensor ({args.data})"
    else:
        X, y, b = make_surrogate(seed=0); src = "FAITHFUL SURROGATE (offline; use --data for real UCI files)"
    d = X.shape[1]
    stream = build_stream(X, y, b, W=3, seed=0)

    print("#" * 92)
    print(f"# REAL-DATA RETENTION TEST  ---  {src}")
    print(f"# {len(X)} samples, d={d}, R={R} classes, {int(b.max())+1} batches, stream={len(stream)}, trials={args.trials}")
    print("#" * 92)
    print(f"\n{'arm':<16}{'post-react acc':>16}{'overall acc':>14}{'SI':>7}{'cov':>7}   type")
    print("-" * 78)
    arms = [("gause_lf", "label-free (ours)"), ("monolith_lf", "label-free"),
            ("learned_div_lf", "label-free"), ("oracle_fixed", "skyline (labels)"),
            ("cl_naive", "supervised CL"), ("cl_ewc", "supervised CL"), ("cl_replay", "supervised CL")]
    rows = {}
    for name, kind in arms:
        K = 3 if name == "monolith_lf" else 1   # monolith is capacity-BOUNDED (K<R)
        a = trials(name, stream, d, n=args.trials, K=K)
        pr, ov, si, cov = a[:, 0].mean(), a[:, 1].mean(), np.nanmean(a[:, 2]), np.nanmean(a[:, 3])
        rows[name] = {"post_react_acc": float(pr), "post_react_sem": sem(a[:, 0]),
                      "overall_acc": float(ov), "si": None if np.isnan(si) else float(si),
                      "coverage": None if np.isnan(cov) else float(cov), "kind": kind}
        print(f"{name:<16}{pr:>16.3f}{ov:>14.3f}"
              f"{('%.2f'%si) if not np.isnan(si) else '   -':>7}"
              f"{('%.2f'%cov) if not np.isnan(cov) else '   -':>7}   {kind}")
    print("-" * 78)
    g, m, o = rows["gause_lf"]["post_react_acc"], rows["monolith_lf"]["post_react_acc"], rows["oracle_fixed"]["post_react_acc"]
    best_cl = max(rows["cl_naive"]["post_react_acc"], rows["cl_ewc"]["post_react_acc"], rows["cl_replay"]["post_react_acc"])
    print(f"\nLabel-free GAUSE post-react acc {g:.3f} vs label-free monolith {m:.3f} (forgets): {(g-m)*100:+.1f} pts.")
    print(f"Oracle skyline (labels) {o:.3f}; GAUSE recovers {g/o*100:.0f}% of it WITHOUT labels.")
    print(f"Best supervised CL baseline (EWC/replay/naive) post-react acc: {best_cl:.3f}.")
    out = Path(__file__).parent / ("gas_real.json" if args.data else "gas_surrogate.json")
    json.dump({"source": src, "n": int(len(X)), "d": int(d), "R": R, "trials": args.trials, "rows": rows},
              open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
