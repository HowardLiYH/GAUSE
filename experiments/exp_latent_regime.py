#!/usr/bin/env python3
"""
Latent-regime (CLASS-INCREMENTAL) GAUSE: removing the oracle regime label.

Everywhere else in the paper, every arm is handed the current regime label r_t
(task-incremental). This experiment drops it. The environment reveals only the
per-method quality of the step; no arm is told which regime is active. The point
is to show the central claim survives in the harder setting: a competitive
population still (a) self-organizes into one-specialist-per-regime measured
against the HIDDEN true regime, and (b) retains dormant niches and reactivates the
correct specialist purely from input similarity, while a capacity-bounded,
usage-driven monolith (also label-free) forgets.

Mechanism. The key realization is that the regime label is not actually needed:
in this environment regime r is signalled by method r being best, and which method
an agent played is OBSERVABLE. So we let each agent carry a niche affinity over the
METHOD space (a proxy for the latent regime) instead of over regimes. An agent
learns "I am the M3 specialist" without ever knowing "this is regime R3." Because
winner-take-all selects on realized quality, the agent whose specialty method
matches the active regime wins -- the winner IS the implicit regime estimate
(Section: latent-regime extension). Reactivation detection is automatic: when a
dormant regime returns, its method works again and the intact specialist wins.

Self-contained (numpy only). The true regime is used ONLY to build the stream and
to score retention -- never fed to any label-free arm.
"""

import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments._affinity_update import eg_eta_for_regimes

PROBE = 15


def make_stream(R=6, W=3, n_epochs=24, steps_per_epoch=80, separation=1.0,
                noise=0.3, seed=0):
    """Same sliding-window non-stationary stream as the retention sweep, but the
    consumer never sees the regime; it only sees the per-method error vector."""
    rng = np.random.default_rng(seed)
    methods = [f"M{i}" for i in range(R)]
    steps, true_regime, reactivated = [], [], []
    prev_active, ever_active = set(), set()
    for e in range(n_epochs):
        active = sorted({(e + j) % R for j in range(W)})
        fresh = set(active) - prev_active
        true_reactivation = fresh & ever_active
        for s in range(steps_per_epoch):
            ri = active[rng.integers(len(active))]
            errs = {methods[mi]: abs(rng.normal(0.0, noise)) + (separation if mi != ri else 0.0)
                    for mi in range(R)}
            steps.append(errs)
            true_regime.append(ri)
            reactivated.append((ri in true_reactivation) and s < PROBE)
        ever_active |= set(active)
        prev_active = set(active)
    return steps, methods, true_regime, reactivated


class LFAgent:
    """Label-free agent: method-success beliefs + a niche affinity over METHODS.
    Never indexed by regime."""
    def __init__(self, methods, K, rng, cap_model="lru"):
        self.methods, self.K, self.rng = methods, K, rng
        self.cap_model = cap_model
        self.store = OrderedDict()  # method -> [alpha, beta]; LRU order
        self.aff = {m: 1.0 / len(methods) + 0.05 * rng.random() for m in methods}
        self._renorm()

    def _renorm(self):
        tot = sum(self.aff.values())
        for m in self.aff:
            self.aff[m] /= tot

    def _ensure(self, m):
        if m in self.store:
            if self.cap_model == "lru":
                self.store.move_to_end(m)
            return
        self.store[m] = [1.0, 1.0]
        if self.cap_model == "lru" and len(self.store) > self.K:
            self.store.popitem(last=False)

    def select(self):
        """Pick a method by Thompson sampling over held beliefs, tilted toward the
        agent's niche affinity (its emerging specialty). No regime label used."""
        scored = {}
        for m in self.methods:
            ab = self.store.get(m, [1.0, 1.0])
            scored[m] = self.rng.beta(*ab) * (0.5 + self.aff[m])
        return max(scored, key=scored.get)

    def learn(self, m, success, q, eta, lam):
        self._ensure(m)
        self.store[m][0 if success else 1] += 1.0
        # EG niche update on the played method (win-only reinforcement).
        self.aff[m] *= np.exp(eta * q)
        self._renorm()


class Monolith:
    """Label-free capacity-bounded monolith: a single belief set over methods with
    K LRU slots, updated every step. It tracks recently useful methods and overwrites
    dormant ones -- catastrophic forgetting with no regime conditioning."""
    def __init__(self, methods, K, rng, cap_model="lru"):
        self.methods, self.K, self.rng, self.cap_model = methods, K, rng, cap_model
        self.store = OrderedDict()

    def _ensure(self, m):
        if m in self.store:
            if self.cap_model == "lru":
                self.store.move_to_end(m)
            return
        self.store[m] = [1.0, 1.0]
        if self.cap_model == "lru" and len(self.store) > self.K:
            self.store.popitem(last=False)

    def select(self):
        scored = {m: self.rng.beta(*self.store.get(m, [1.0, 1.0])) for m in self.methods}
        return max(scored, key=scored.get)

    def learn(self, m, success):
        self._ensure(m)
        self.store[m][0 if success else 1] += 1.0


def quality(errs, m):
    mx = max(errs.values()); span = (mx - min(errs.values())) or 1.0
    return (mx - errs[m]) / span


def run_population(steps, methods, true_regime, react, K, lam=1.0, seed=0,
                   cap_model="lru", label_free=True):
    rng = np.random.default_rng(seed)
    R = len(methods)
    eta = eg_eta_for_regimes(R)
    agents = [LFAgent(methods, K, rng, cap_model=cap_model) for _ in range(R)]
    err_all, err_re = [], []
    win_by_regime = {r: np.zeros(R) for r in range(R)}  # diagnostics: agent wins per true regime
    for errs, r, is_re in zip(steps, true_regime, react):
        picks = [a.select() for a in agents]
        qs = [quality(errs, m) for m in picks]
        if lam > 0:
            scores = [qs[i] + lam * (agents[i].aff[picks[i]] - 1.0 / R) for i in range(R)]
        else:
            scores = qs
        w = int(np.argmax(scores))
        agents[w].learn(picks[w], success=qs[w] >= 0.5, q=qs[w], eta=eta, lam=lam)
        win_by_regime[r][w] += 1
        served_err = errs[picks[w]]  # winner serves; no label used to choose server
        err_all.append(served_err)
        if is_re:
            err_re.append(served_err)
    half = len(err_all) // 2
    # specialization: each agent's win-distribution over regimes -> how peaked (SI)
    return (float(np.mean(err_all[half:])),
            float(np.mean(err_re) if err_re else np.nan),
            win_by_regime)


def run_monolith(steps, methods, true_regime, react, K, seed=0, cap_model="lru"):
    rng = np.random.default_rng(seed)
    mono = Monolith(methods, K, rng, cap_model=cap_model)
    err_all, err_re = [], []
    for errs, r, is_re in zip(steps, true_regime, react):
        m = mono.select()
        q = quality(errs, m)
        mono.learn(m, success=q >= 0.5)
        e = errs[m]
        err_all.append(e)
        if is_re:
            err_re.append(e)
    half = len(err_all) // 2
    return float(np.mean(err_all[half:])), float(np.mean(err_re) if err_re else np.nan)


def specialization_index(win_by_regime, R):
    """Mean over true regimes of (1 - normalized entropy of the winning-agent
    distribution). 1 = exactly one agent wins each regime (clean latent recovery)."""
    sis = []
    for r in range(R):
        w = win_by_regime[r]
        tot = w.sum()
        if tot == 0:
            continue
        p = w / tot
        p = p[p > 0]
        H = -np.sum(p * np.log(p))
        sis.append(1.0 - H / np.log(R))
    return float(np.mean(sis)) if sis else 0.0


def coverage(win_by_regime, R):
    """Fraction of true regimes that have a clear majority specialist (>50% of wins)."""
    owners = set()
    covered = 0
    for r in range(R):
        w = win_by_regime[r]
        if w.sum() == 0:
            continue
        top = int(np.argmax(w))
        if w[top] / w.sum() > 0.5:
            covered += 1
            owners.add(top)
    return covered / R, len(owners)


def trials(fn, n_trials=20, **kw):
    return [fn(seed=s, **kw) for s in range(n_trials)]


def _m(xs):
    xs = np.asarray([x for x in xs if not np.isnan(x)], float)
    return float(xs.mean()), float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else 0.0


def main():
    R, W = 6, 3
    n_trials = 20
    steps, methods, true_regime, react = make_stream(R=R, W=W, seed=0)
    print("#" * 92)
    print("# LATENT-REGIME (class-incremental) GAUSE: regime label REMOVED.")
    print("# Every arm sees only the per-method quality, never the regime id.")
    print(f"# Stream R={R}, W={W}, {len(steps)} steps, n_trials={n_trials}, hard LRU.")
    print("#" * 92)

    print(f"\n{'K':>3}{'LF-GAUSE react':>18}{'LF-monolith react':>20}{'LF-GAUSE SI':>14}{'coverage':>12}")
    print("-" * 70)
    rows = []
    for K in range(1, R + 1):
        pop = trials(run_population, n_trials=n_trials, steps=steps, methods=methods,
                     true_regime=true_regime, react=react, K=K, lam=1.0)
        g_over = [p[0] for p in pop]; g_re = [p[1] for p in pop]
        sis = [specialization_index(p[2], R) for p in pop]
        covs = [coverage(p[2], R)[0] for p in pop]
        mono = trials(run_monolith, n_trials=n_trials, steps=steps, methods=methods,
                      true_regime=true_regime, react=react, K=K)
        m_re = [m[1] for m in mono]; m_over = [m[0] for m in mono]
        gre_m, gre_s = _m(g_re); mre_m, mre_s = _m(m_re)
        si_m, _ = _m(sis); cov_m, _ = _m(covs)
        rows.append({"K": K,
                     "lf_gause_overall": _m(g_over)[0], "lf_gause_react": gre_m, "lf_gause_react_sem": gre_s,
                     "lf_monolith_overall": _m(m_over)[0], "lf_monolith_react": mre_m, "lf_monolith_react_sem": mre_s,
                     "lf_gause_SI": si_m, "lf_gause_coverage": cov_m})
        print(f"{K:>3}{gre_m:>18.3f}{mre_m:>20.3f}{si_m:>14.3f}{cov_m:>12.2f}")
    print("-" * 70)

    r1 = next(r for r in rows if r["K"] == 1)
    print(f"\nLabel-FREE GAUSE at K=1: post-reactivation error {r1['lf_gause_react']:.3f} "
          f"(+/-{1.96*r1['lf_gause_react_sem']:.3f}); SI {r1['lf_gause_SI']:.2f}, "
          f"coverage {r1['lf_gause_coverage']:.2f}.")
    print(f"Label-FREE monolith at K=1: post-reactivation error {r1['lf_monolith_react']:.3f} "
          f"(+/-{1.96*r1['lf_monolith_react_sem']:.3f}) -- forgets.")
    print(f"For reference, the LABEL-GIVEN (task-incremental) GAUSE was 0.283 at K=1.")
    gap = (r1['lf_monolith_react'] - r1['lf_gause_react']) / r1['lf_monolith_react'] * 100
    print(f"Label-free GAUSE beats the label-free monolith on retention by {gap:+.1f}% at K=1.")

    out_dir = Path(__file__).parent.parent / "results" / "nonstationary_capacity"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results_latent_regime.json", "w", encoding="utf-8") as f:
        json.dump({"R": R, "W": W, "n_trials": n_trials, "rows": rows,
                   "label_given_gause_react_K1": 0.283}, f, indent=2)
    print(f"\nSaved -> {out_dir / 'results_latent_regime.json'}")
    return rows


if __name__ == "__main__":
    main()
