#!/usr/bin/env python3
"""
Route B: couple niche specialization to behavior, then test whether it rescues
the strong thesis.

The current prediction pipeline decouples niche affinity from performance:
method selection samples from per-regime method beliefs, and the winner is the
agent with the lowest error -- niche affinity is a dead-end readout that never
feeds back into behavior. Route B re-couples them:

  1. WINNER DETERMINATION uses a coupled score:
         score_i = relative_quality_i + lambda * (affinity_i[regime] - 1/R)
     so an agent that has specialized into the current regime gets a leg up,
     exactly like the headline NichePopulation -- but here relative_quality is
     REAL prediction quality, not a random draw.
  2. WINNER-TAKE-ALL LEARNING: only the winner updates its method belief for the
     current regime, so the regime's specialist accumulates clean expertise there
     while generalists do not.
  3. READOUT BY ROUTING: the population's prediction for a regime is taken from
     the agent whose primary niche == that regime (the "responsible specialist").

This makes specialization causally relevant: if regimes genuinely favor different
methods, routing to the regime's specialist should beat the shuffled control.

We run TWO conditions to separate "algorithm works" from "data has structure":
  - SYNTHETIC regime-champion task: method m is best in regime m by construction.
    Expectation: real >> shuffled  => the MECHANISM captures real structure.
  - REAL domains (energy/weather/finance): oracle ceiling is ~0-1.9%.
    Expectation: real ~= shuffled  => no exploitable structure to capture,
    consistent with the oracle gate (exp_route_b_diagnostic.py).
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.exp_lambda_zero_real import load_domain_data, get_domain_predictor
from experiments._affinity_update import eg_eta_for_regimes

# A "step" is (regime_label, {method: error}). Building these once lets the
# coupled population be agnostic to where the errors come from.
Step = Tuple[str, Dict[str, float]]


# --------------------------------------------------------------------------- #
# Data providers
# --------------------------------------------------------------------------- #
def make_real_steps(domain: str) -> Tuple[List[Step], List[str], List[str]]:
    """Per-step (regime, method->error) from real domain predictors."""
    values, regimes, methods = load_domain_data(domain)
    steps: List[Step] = []
    for idx in range(20, len(values)):
        history = values[:idx]
        true_val = values[idx]
        errs = {m: abs(get_domain_predictor(domain, m, history, idx) - true_val)
                for m in methods}
        steps.append((regimes[idx], errs))
    regime_labels = sorted({r for r, _ in steps})
    return steps, regime_labels, methods


def make_synthetic_steps(n_regimes: int = 4, n_steps: int = 1500,
                         separation: float = 1.0, noise: float = 0.3,
                         seed: int = 0) -> Tuple[List[Step], List[str], List[str]]:
    """Regime-champion task: method m is the unique best method in regime m.

    Error of method m in regime r is ``noise_draw + separation * (m != r)``.
    With separation > noise the per-regime optimum is unambiguous and the
    specialist oracle strictly beats any single method.
    """
    rng = np.random.default_rng(seed)
    regime_labels = [f"R{i}" for i in range(n_regimes)]
    methods = [f"M{i}" for i in range(n_regimes)]
    steps: List[Step] = []
    for _ in range(n_steps):
        r = rng.integers(n_regimes)
        errs = {}
        for mi, m in enumerate(methods):
            base = abs(rng.normal(0.0, noise))
            errs[m] = base + (separation if mi != r else 0.0)
        steps.append((regime_labels[r], errs))
    return steps, regime_labels, methods


# --------------------------------------------------------------------------- #
# Coupled population
# --------------------------------------------------------------------------- #
class CoupledAgent:
    def __init__(self, agent_id: int, regimes: List[str], methods: List[str], rng):
        self.agent_id = agent_id
        self.regimes = regimes
        self.methods = methods
        self.rng = rng
        # Per-regime method beliefs (Beta over "this method did well here").
        self.beliefs = {r: {m: [1.0, 1.0] for m in methods} for r in regimes}
        # Niche affinities (probability simplex over regimes).
        self.affinity = {r: 1.0 / len(regimes) for r in regimes}

    def select_method(self, regime: str) -> str:
        samples = {m: self.rng.beta(a, b) for m, (a, b) in self.beliefs[regime].items()}
        return max(samples, key=samples.get)

    def primary_niche(self) -> str:
        return max(self.affinity, key=self.affinity.get)

    def update_belief(self, regime: str, method: str, success: bool):
        if success:
            self.beliefs[regime][method][0] += 1.0
        else:
            self.beliefs[regime][method][1] += 1.0

    def update_affinity(self, regime: str, reward: float, eta: float):
        # Exponentiated-gradient step toward the rewarded regime, then renormalize.
        self.affinity[regime] *= np.exp(eta * reward)
        total = sum(self.affinity.values())
        for r in self.affinity:
            self.affinity[r] /= total


class CoupledPopulation:
    def __init__(self, regimes: List[str], methods: List[str], n_agents: int = 6,
                 lam: float = 0.3, seed: int = 0, coupled: bool = True):
        self.regimes = regimes
        self.methods = methods
        self.lam = lam
        # When coupled=False the niche affinity is still tracked (and routing still
        # reads it out) but it does NOT enter winner determination -- this is the
        # current/decoupled pipeline. The ablation isolates exactly this term.
        self.coupled = coupled
        self.rng = np.random.default_rng(seed)
        self.eta = eg_eta_for_regimes(len(regimes))
        self.agents = [CoupledAgent(i, regimes, methods, self.rng) for i in range(n_agents)]
        self.inv_r = 1.0 / len(regimes)
        self.routed_errors: List[float] = []

    def run(self, steps: List[Step]):
        for regime, errs in steps:
            max_e = max(errs.values())
            min_e = min(errs.values())
            span = (max_e - min_e) or 1.0

            chosen = {}
            quality = {}
            for ag in self.agents:
                m = ag.select_method(regime)
                chosen[ag.agent_id] = m
                quality[ag.agent_id] = (max_e - errs[m]) / span  # in [0,1], higher=better

            # Coupled score: real quality + niche leg-up for the regime's specialists.
            # Decoupled (ablation): pure performance, niche affinity does not vote.
            niche_term = self.lam if self.coupled else 0.0
            scores = {ag.agent_id: quality[ag.agent_id]
                      + niche_term * (ag.affinity[regime] - self.inv_r)
                      for ag in self.agents}
            winner_id = max(scores, key=scores.get)

            for ag in self.agents:
                if ag.agent_id == winner_id:
                    q = quality[ag.agent_id]
                    ag.update_belief(regime, chosen[ag.agent_id], success=q >= 0.5)
                    ag.update_affinity(regime, reward=q, eta=self.eta)

            # Readout: route to the responsible specialist(s) for this regime.
            specialists = [ag for ag in self.agents if ag.primary_niche() == regime]
            if specialists:
                routed_err = np.mean([errs[ag.select_method(regime)] for ag in specialists])
            else:
                routed_err = errs[chosen[winner_id]]
            self.routed_errors.append(routed_err)

    def steady_error(self) -> float:
        half = len(self.routed_errors) // 2
        return float(np.mean(self.routed_errors[half:]))

    def specialization_index(self) -> float:
        # Mean Herfindahl concentration of agents' affinity over regimes,
        # normalized so uniform -> 0 and one-hot -> 1.
        R = len(self.regimes)
        sis = []
        for ag in self.agents:
            h = sum(a * a for a in ag.affinity.values())
            sis.append((h - 1.0 / R) / (1.0 - 1.0 / R))
        return float(np.mean(sis))

    def population_diversity(self) -> float:
        """Coverage = fraction of regimes that are some agent's primary niche.

        This is the Route A (division-of-labor) metric: it asks whether agents
        spread across DIFFERENT niches, independent of whether that helps task
        performance. 1.0 means every regime is claimed by at least one agent;
        low values mean the population piled onto a few niches.
        """
        claimed = {ag.primary_niche() for ag in self.agents}
        return len(claimed) / len(self.regimes)


# --------------------------------------------------------------------------- #
# Experiment driver
# --------------------------------------------------------------------------- #
def run_condition(steps: List[Step], regimes: List[str], methods: List[str],
                 shuffle: bool, n_trials: int, base_seed: int,
                 coupled: bool = True) -> Dict:
    si_vals, err_vals, div_vals = [], [], []
    for t in range(n_trials):
        if shuffle:
            # Permute regime labels -> destroys regime<->error alignment.
            perm_rng = np.random.default_rng(base_seed + 1000 + t)
            labels = [s[0] for s in steps]
            shuffled_labels = list(perm_rng.permutation(labels))
            cond_steps = [(shuffled_labels[i], steps[i][1]) for i in range(len(steps))]
        else:
            cond_steps = steps
        pop = CoupledPopulation(regimes, methods, seed=base_seed + t, coupled=coupled)
        pop.run(cond_steps)
        si_vals.append(pop.specialization_index())
        err_vals.append(pop.steady_error())
        div_vals.append(pop.population_diversity())
    return {"si": np.array(si_vals), "err": np.array(err_vals),
            "div": np.array(div_vals)}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp else 0.0


def evaluate(name: str, steps, regimes, methods, n_trials: int, coupled: bool = True):
    real = run_condition(steps, regimes, methods, shuffle=False, n_trials=n_trials,
                         base_seed=1, coupled=coupled)
    shuf = run_condition(steps, regimes, methods, shuffle=True,
                         n_trials=max(1, n_trials), base_seed=2, coupled=coupled)
    # One-sided: real error LOWER than shuffled error => routing to specialists helps.
    _t, p = stats.ttest_ind(real["err"], shuf["err"], alternative="less")
    gap_pct = (shuf["err"].mean() - real["err"].mean()) / shuf["err"].mean() * 100
    d = cohens_d(real["err"], shuf["err"])

    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")
    print(f"  SI (real):       {real['si'].mean():.3f} +/- {real['si'].std():.3f}")
    print(f"  SI (shuffled):   {shuf['si'].mean():.3f} +/- {shuf['si'].std():.3f}")
    print(f"  niche coverage (real/shuf):  {real['div'].mean():.3f} / {shuf['div'].mean():.3f}"
          f"   [Route A division-of-labor metric]")
    print(f"  routed err real:     {real['err'].mean():.4f} +/- {real['err'].std():.4f}")
    print(f"  routed err shuffled: {shuf['err'].mean():.4f} +/- {shuf['err'].std():.4f}")
    print(f"  performance gap (real better by): {gap_pct:+.2f}%")
    print(f"  one-sided p (real<shuf): {p:.4g}   Cohen's d: {d:.3f}")
    verdict = "MEANINGFUL (real beats shuffled)" if (p < 0.05 and gap_pct > 1.0) \
        else "NOT meaningful (real ~= shuffled)"
    print(f"  => {verdict}")
    return {"name": name, "gap_pct": gap_pct, "p": p, "d": d,
            "si_real": real["si"].mean(), "si_shuf": shuf["si"].mean()}


def main():
    print("#" * 80)
    print("# ROUTE B: coupled architecture -- does coupling niche->behavior rescue")
    print("# the strong (environment-driven) thesis?")
    print("#" * 80)

    results = []

    # 1) Synthetic regime-champion task: structure exists by construction.
    syn_steps, syn_regimes, syn_methods = make_synthetic_steps(
        n_regimes=4, n_steps=1500, separation=1.0, noise=0.3, seed=0)
    results.append(evaluate("SYNTHETIC regime-champion (structure by construction)",
                            syn_steps, syn_regimes, syn_methods, n_trials=20))

    # 2) Real domains: oracle ceiling ~0-1.9%.
    for domain in ["energy", "weather", "finance"]:
        steps, regimes, methods = make_real_steps(domain)
        results.append(evaluate(f"REAL: {domain}", steps, regimes, methods, n_trials=20))

    print("\n" + "#" * 80)
    print("# SUMMARY")
    print("#" * 80)
    print(f"{'condition':<48}{'gap%':>9}{'p':>10}{'d':>8}")
    for r in results:
        print(f"{r['name'][:46]:<48}{r['gap_pct']:>9.2f}{r['p']:>10.3g}{r['d']:>8.2f}")
    print("\nReading: a large positive gap with p<0.05 on the SYNTHETIC task shows the")
    print("coupled mechanism captures regime structure WHEN IT EXISTS. Near-zero gaps")
    print("on the REAL domains then reflect the data (oracle ceiling ~0-1.9%), not a")
    print("broken algorithm. If even SYNTHETIC fails, the architecture itself is at fault.")


if __name__ == "__main__":
    main()
