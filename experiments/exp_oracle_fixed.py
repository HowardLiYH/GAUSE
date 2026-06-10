#!/usr/bin/env python3
"""
Oracle fixed-assignment baseline (the honest competitor for the retention result).

Reviewer concern: since the retention experiment hands every arm the current
regime label r_t (task-incremental setting), the *trivial* reward-independent,
fully-covering baseline is not "random diversity" but an ORACLE FIXED ASSIGNMENT:
pin agent i to a covering niche by hand (a permutation at K=1), with no learning
of WHERE to specialize. That arm is reward-independent and covering by
construction, so Principle 1 predicts it should retain in full.

This script adds that arm to the non-stationary retention sweep and compares it
to GAUSE (emergent competition). The point is NOT that one beats the other: it is
that GAUSE *matches* the hand-assigned oracle WITHOUT being given the assignment --
competition discovers the covering permutation for free. Any residual gap measures
the price (or bonus) of discovering vs. being told the assignment.

Self-contained (numpy only; no scipy) so it reproduces the published GAUSE/monolith
numbers as a cross-check while adding the oracle_fixed arm.
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments._affinity_update import eg_eta_for_regimes

PROBE = 15
_SOFT_EPS = 0.5
_SOFT_INTER = 0.04


def make_nonstationary_stream(R=6, W=3, n_epochs=24, steps_per_epoch=80,
                              separation=1.0, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    regimes = [f"R{i}" for i in range(R)]
    methods = [f"M{i}" for i in range(R)]
    steps, reactivated = [], []
    prev_active, ever_active = set(), set()
    for e in range(n_epochs):
        active = sorted({(e + j) % R for j in range(W)})
        fresh = set(active) - prev_active
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


class StoreAgent:
    def __init__(self, regimes, methods, K, rng, fixed=None, cap_model="lru"):
        self.regimes, self.methods, self.K, self.rng = regimes, methods, K, rng
        self.cap_model = cap_model
        self.store = OrderedDict()
        self.fixed = fixed is not None
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
        return sum(ab[0] + ab[1] - 2.0 for ab in self.store[r].values())

    def _interfere(self, current):
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
        if r in self.store:
            if self.cap_model == "lru":
                self.store.move_to_end(r)
            return False
        if self.fixed:
            return False
        self.store[r] = {m: [1.0, 1.0] for m in self.methods}
        if self.cap_model == "lru" and len(self.store) > self.K:
            self.store.popitem(last=False)
        return True

    def select(self, r):
        if r in self.store:
            samples = {m: self.rng.beta(*self.store[r][m]) for m in self.methods}
            return max(samples, key=samples.get)
        return self.methods[self.rng.integers(len(self.methods))]

    def update_belief(self, r, m, success):
        if r in self.store:
            self.store[r][m][0 if success else 1] += 1.0
            if self.cap_model == "soft":
                self._interfere(r)

    def update_affinity(self, r, reward, eta):
        self.affinity[r] *= np.exp(eta * reward)
        self._renorm()


def _covering_assignment(R, K, i):
    """Oracle covering niche for agent i: the K regimes starting at i (cyclically).
    At K=1 this is the identity permutation (agent i <-> regime i): perfect cover,
    no collisions, no gaps -- the strongest hand-assigned reward-independent arm."""
    return [(i + j) % R for j in range(K)]


def run(steps, react, regimes, methods, K, mode, lam=1.5, beta=1.0,
        gate_lr=0.2, eps_route=0.1, cap_model="lru", seed=0):
    rng = np.random.default_rng(seed)
    eta = eg_eta_for_regimes(len(regimes))
    R = len(regimes)
    N = R
    gate = {r: np.zeros(N) for r in regimes}

    if mode == "monolith":
        agents = [StoreAgent(regimes, methods, K, rng, cap_model=cap_model)]
    elif mode == "random":
        agents = [StoreAgent(regimes, methods, K, rng, cap_model=cap_model,
                             fixed=[regimes[i] for i in rng.choice(R, size=K, replace=False)])
                  for _ in range(N)]
    elif mode == "oracle_fixed":
        agents = [StoreAgent(regimes, methods, K, rng, cap_model=cap_model,
                             fixed=[regimes[k] for k in _covering_assignment(R, K, i)])
                  for i in range(N)]
    else:
        agents = [StoreAgent(regimes, methods, K, rng, cap_model=cap_model) for _ in range(N)]

    err_all, err_react = [], []
    for (regime, errs), is_react in zip(steps, react):
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
            e = int(rng.integers(N)) if rng.random() < eps_route else int(np.argmax(gate[regime]))
            expert = agents[e]
            expert.ensure(regime)
            expert.update_belief(regime, chosen[id(expert)], success=quality[id(expert)] >= 0.5)
            expert.update_affinity(regime, quality[id(expert)], eta)
            gate[regime][e] += gate_lr * (quality[id(expert)] - 0.5)
            w = None
        elif mode in ("random", "oracle_fixed"):
            # Reward-independent fixed niches: no competition, no affinity learning;
            # each agent only refreshes beliefs for regimes it already owns.
            for a in agents:
                if a.competent(regime):
                    a.update_belief(regime, chosen[id(a)], success=quality[id(a)] >= 0.5)
            w = None
        elif mode == "diversity":
            denom = sum(a.affinity[regime] for a in agents) or 1.0
            owner = max(agents, key=lambda a: a.affinity[regime])
            for a in agents:
                p_i = a.affinity[regime] / denom
                a.update_affinity(regime, quality[id(a)] + beta * np.log(p_i + 1e-9), eta)
            owner.ensure(regime)
            owner.update_belief(regime, chosen[id(owner)], success=quality[id(owner)] >= 0.5)
            w = None
        else:  # specialized = GAUSE
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

        if mode == "moe":
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


def trials(steps, react, regimes, methods, K, mode, n_trials=20, **kw):
    out = [run(steps, react, regimes, methods, K, mode, seed=s, **kw) for s in range(n_trials)]
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def _sem(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def welch(a, b):
    """Welch's t statistic + normal-approx two-sided p (n>=20 -> fine).
    Returns (t, p_approx, dof)."""
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    b = np.asarray(b, float); b = b[~np.isnan(b)]
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 1.0
    t = (a.mean() - b.mean()) / se
    # normal approximation to the two-sided p-value
    from math import erf, sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0))))
    return float(t), float(p)


def main():
    R, W = 6, 3
    n_trials = 20
    steps, regimes, methods, react = make_nonstationary_stream(R=R, W=W, seed=0)
    print("#" * 92)
    print("# ORACLE FIXED-ASSIGNMENT vs GAUSE (emergent competition) -- retention under")
    print("# non-stationarity. Hard LRU capacity model, R=6, W=3, n_trials=20.")
    print("#" * 92)
    print(f"\n{'K':>3}{'monolith':>14}{'MoE':>14}{'random':>14}{'oracle_fix':>14}"
          f"{'learned_div':>14}{'GAUSE':>14}   (post-reactivation error)")
    print("-" * 100)
    rows = []
    for K in range(1, R + 1):
        _, mr = trials(steps, react, regimes, methods, K, "monolith", n_trials=n_trials)
        _, gr = trials(steps, react, regimes, methods, K, "moe", n_trials=n_trials)
        _, rr = trials(steps, react, regimes, methods, K, "random", n_trials=n_trials)
        _, orr = trials(steps, react, regimes, methods, K, "oracle_fixed", n_trials=n_trials)
        _, dr = trials(steps, react, regimes, methods, K, "diversity", n_trials=n_trials)
        _, sr = trials(steps, react, regimes, methods, K, "specialized", n_trials=n_trials)
        t_go, p_go = welch(sr, orr)  # GAUSE vs oracle_fixed
        rows.append({"K": K,
                     "monolith_react": float(np.nanmean(mr)), "monolith_react_sem": _sem(mr),
                     "moe_react": float(np.nanmean(gr)), "moe_react_sem": _sem(gr),
                     "random_react": float(np.nanmean(rr)), "random_react_sem": _sem(rr),
                     "oracle_fixed_react": float(np.nanmean(orr)), "oracle_fixed_react_sem": _sem(orr),
                     "diversity_react": float(np.nanmean(dr)), "diversity_react_sem": _sem(dr),
                     "gause_react": float(np.nanmean(sr)), "gause_react_sem": _sem(sr),
                     "gause_vs_oracle_t": t_go, "gause_vs_oracle_p": p_go})
        print(f"{K:>3}{np.nanmean(mr):>14.3f}{np.nanmean(gr):>14.3f}{np.nanmean(rr):>14.3f}"
              f"{np.nanmean(orr):>14.3f}{np.nanmean(dr):>14.3f}{np.nanmean(sr):>14.3f}")
    print("-" * 100)
    print("\nKey comparison (GAUSE = discovered assignment; oracle_fixed = hand-given covering permutation):")
    for K in (1, 3):
        r = next(x for x in rows if x["K"] == K)
        diff = r["gause_react"] - r["oracle_fixed_react"]
        print(f"  K={K}: GAUSE {r['gause_react']:.3f}+/-{1.96*r['gause_react_sem']:.3f}  vs  "
              f"oracle_fixed {r['oracle_fixed_react']:.3f}+/-{1.96*r['oracle_fixed_react_sem']:.3f}   "
              f"(diff {diff:+.3f}, Welch p={r['gause_vs_oracle_p']:.2f})")
    out_dir = Path(__file__).parent.parent / "results" / "nonstationary_capacity"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results_oracle_fixed.json", "w", encoding="utf-8") as f:
        json.dump({"R": R, "W": W, "probe": PROBE, "n_trials": n_trials, "rows": rows}, f, indent=2)
    print(f"\nSaved -> {out_dir / 'results_oracle_fixed.json'}")
    return rows


if __name__ == "__main__":
    main()
