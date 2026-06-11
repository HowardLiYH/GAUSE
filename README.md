# 🧬 GAUSE: Emergent Specialization in Learner Populations

### Generalist-averse Affinity Update for Specialist Emergence

<div align="center">

<img src="assets/cover.jpeg" alt="Emergent Specialization - 6 Domain Learners" width="100%">

<br><br>

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data: 100% Real](https://img.shields.io/badge/Data-100%25%20Real-green.svg)](#data)

**Niche Partitioning Without Explicit Coordination**

[Paper](#paper) • [Installation](#installation) • [Quick Start](#quick-start) • [Experiments](#experiments) • [Results](#key-results) • [Citation](#citation)

</div>

---

## 📖 Abstract

**GAUSE** (Generalist-averse Affinity Update for Specialist Emergence)<sup>†</sup> is a population-based learning system where learners **spontaneously specialize** to different environmental regimes without explicit supervision. Drawing from ecological niche theory, it couples **competitive exclusion with niche affinity** to create evolutionary pressure for strategy-space partitioning.

<sub>† The name also honors G. F. Gause, whose competitive exclusion principle (1934) the mechanism operationalizes.</sub>

**Core Thesis (v4.1):** Under bounded per-agent capacity and non-stationarity, **retention of dormant-regime knowledge tracks a single property: whether capacity assignment is independent of current task reward.** Reward-chasing allocations (a capacity-bounded monolith, a learned Mixture-of-Experts router) forget dormant regimes and relearn them on reactivation; reward-independent assignments (fixed random niches, an EOI/CDS-style intrinsic-diversity objective, converged competitive exclusion) retain them. Competition is the most **parsimonious** route to reward-independent specialization: no gate to train, no diversity objective to tune, no freezing schedule to pick — the assignment is the equilibrium the dynamics converge to. Supporting claim (unchanged): competition alone, without explicit diversity incentives, suffices to induce emergent specialization (mean SI = 0.65 at λ = 0).

> **Note on Terminology:** We use "learner" to denote individual units in the population, each implementing Thompson Sampling over prediction methods. This distinguishes our approach from LLM-based "agents" which are autoregressive language models.

**Validated on 6 domains (100% REAL DATA):**
- 📈 **Crypto** - Bybit Exchange (44,000+ bars) ✅ Real
- 📊 **Commodities** - FRED US Government (5,630 daily prices) ✅ Real
- 🌤️ **Weather** - Open-Meteo (9,105 observations) ✅ Real
- ☀️ **Solar** - Open-Meteo Satellite (116,834 hourly) ✅ Real
- 🚕 **Traffic** - NYC TLC Yellow Taxi (2,879 hourly trips) ✅ Real
- 🌬️ **Air Quality** - Open-Meteo PM2.5 (2,880 hourly readings) ✅ Real

---

## 🎯 Key Results (v4.3 — real-data validated)

> **Changelog (v4.2.0, June 2026):** Addressed three reviewer concerns and added the
> **oracle fixed-assignment skyline** (GAUSE matches it without the assignment:
> within 0.030 at K=1, indistinguishable for K≥3 and at every K under the soft model),
> a proved **reallocation-rate proposition** (replacing the hand-wave necessity
> claim), and a **class-incremental (label-free)** variant — the retention result
> survives dropping the regime label (SI 0.75, post-react 0.38 vs. 1.07 for a
> label-free monolith). New tooling: `experiments/exp_oracle_fixed.py`,
> `exp_latent_regime.py`, `exp_real_data_gas.py` (UCI Gas Sensor Array Drift, real
> data, `--data PATH`), `exp_robustness_sweep.py`. Explainer tightened (soft-model,
> population-sizing, and per-domain SI moved to an appendix). Full detail in
> [`CHANGELOG.md`](CHANGELOG.md).
>
> **Changelog (v4.3.0, June 2026) — REAL DATA, the honest split.** Ran the full
> retention + label-free + CL pipeline on the real UCI Gas Sensor Array Drift stream
> (13,910 samples, 128-d, 6 classes, 10 batches / 36 months). The result moves the
> contribution **off the performance axis and onto the mechanism/lens axis**, and we
> report it straight: **(1)** a full-capacity online linear classifier does *not*
> catastrophically forget on real gas features (naive/replay 0.68 post-react, 0.93
> overall; EWC *worse* at 0.44 because its anchor fights drift) — so the
> forgetting dissociation is *relocated* to representation-sharing **neural** experts
> (permuted-digits: router 0.576 vs GAUSE 0.218, +62%), not arbitrary streams;
> **(2)** label-free recovery **collapses** under real overlap (GAUSE-LF post-react
> 0.00, SI 0.53, coverage 0.83; the label-free monolith is *also* 0.00, so the
> dissociation vanishes) — the synthetic 1:1 method↔regime signature was load-bearing,
> bounding the class-incremental claim to *separable-signal regimes*; **(3)** framing
> confirmed: on real data this is a mechanism/parsimony/lens paper, not a performance
> paper. Full numbers, figure (`paper/figures/fig_real_data_gas.pdf`), and per-concern
> verdict in [`CHANGELOG.md`](CHANGELOG.md) (v4.3.0).

> **Note (v4.0.0):** As of June 2026, the niche affinity update has been
> upgraded from the V3 additive heuristic to the canonical
> **exponentiated-gradient (Hedge / multiplicative-weights)** update.
> The numbers below are V4 (rescaled-η). For the V3-era numbers and the
> rationale for the transition, see [`CHANGELOG.md`](CHANGELOG.md) and
> [`docs/V4_FINAL_REPORT.md`](docs/V4_FINAL_REPORT.md). All qualitative
> findings are preserved; quantitative magnitudes are substantially
> strengthened.

### ⭐ Headline (v4.1): Coverage and Retention Under Bounded Capacity

Five capacity-allocation mechanisms compared at matched per-agent capacity K (each agent can master only K of R regimes), on a non-stationary stream where regimes go dormant and reactivate:

| Arm (K=1, hard LRU model) | Assignment signal | Post-reactivation error | Retains dormant regimes? |
|---|---|---|---|
| Monolith (capacity K) | current active regime | 1.015 | ❌ |
| MoE learned router | current task reward | 0.928 | ❌ |
| Random fixed niches | none (frozen) | 0.603 | ⚠️ (coverage gaps) |
| EOI/CDS-style diversity | intrinsic identity reward | 0.322 | ✅ |
| **GAUSE (ours)** | converged identity | **0.283** | ✅ |

- **Retention tracks reward-independence five-for-five.** The specialized population beats the capacity-matched monolith by **+33% overall / +71% post-reactivation** (p < 10⁻³⁶ at K=3); the reward-driven router fails like the monolith (p ~ 10⁻³⁵ vs. ours at K=1).
- **Why:** a dormant regime emits no reward gradient, so a reward-driven gate gets **no signal** to reserve capacity for it (idealized Observation in the paper). A converged specialist simply idles through dormancy and retains its niche structurally.
- **Robust to the memory model:** the same dissociation holds when hard LRU eviction is replaced by soft interference decay (`--soft`; monolith forgets at 0.85–0.92, ours retains at ≈0.25, p ~ 10⁻⁴⁰).
- **Spatial coverage is a commodity** — competition, explicit diversity, and the router all achieve it (competition wins +57% synthetic / +8.7% traffic at K=1 over *random* diversity but only matches purpose-built baselines for K≥2). Competition's value there is parsimony, not dominance.
- **Corroborates MoE continual-learning theory** (ICLR'25, arXiv:2406.16437): their proof that the gate must be *frozen* for CL convergence is, in our terms, making the assignment reward-independent — competition reaches that state emergently.

Reproduce: `python experiments/exp_capacity_division.py` and `python experiments/exp_nonstationary_capacity.py --soft`.

### 🧪 Closing the Design Space (v4.1): Function Approximation, a Hybrid Router, Drift, Sizing, and the Label-Free Variant

Six further experiments stress-test the reward-independence story across architectures, fixes, failure modes, and the harder dropped-label setting:

| Experiment | Script | Headline finding |
|---|---|---|
| Function-approx CL (gradient-trained MLP experts, permuted-digits, R=5) | `exp_function_approx_cl.py` | Router forgetting is **architecture-agnostic**: at E=R, GAUSE post-reactivation test error 0.218 vs MoE router 0.576 (**+62% lower**, p ~ 10⁻⁹); the router does not improve with more experts |
| Hybrid router **+ reservation term** | `exp_hybrid_router.py` | Reservation recovers most retention at K≥2 (**+48% at K=3**, p ~ 10⁻¹¹) but **cannot help at K=1** (no spare slot); GAUSE retains even at K=1 — reward-independence of *protection* is the operative property |
| Intra-regime concept drift | `exp_intra_regime_drift.py` | **Stale specialists** quantified: standard GAUSE overall error 0.90 (worse than a relearning K=3 monolith, 0.35); a lightweight staleness trigger recovers most of it (0.43, ~53% lower) |
| Off-diagonal population sizing | `exp_population_sizing.py` | Scarce agents (N<R) under-cover, governed by **N not N·K**; surplus agents (N≫R) idle harmlessly (redundant = N−R). Rule: **provision N ≳ R** |
| Oracle fixed-assignment skyline | `exp_oracle_fixed.py` | GAUSE recovers the hand-assigned skyline **without being given it**: within **0.030** at K=1 (0.283 vs 0.253 floor), statistically indistinguishable by K=3 |
| Class-incremental (**label-free**) GAUSE | `exp_latent_regime.py` | The central retention result **survives dropping the regime label**: SI 0.75, coverage 0.86, post-reactivation **0.38 vs 1.07** for a label-free forgetting monolith (+65%), reactivation detected from input similarity alone |

Reproduce: `python experiments/exp_function_approx_cl.py` (needs `torch`), then `exp_hybrid_router.py`, `exp_intra_regime_drift.py`, `exp_population_sizing.py`, `exp_oracle_fixed.py`, `exp_latent_regime.py`.

### Cross-Domain Experimental Results (Unified Pipeline — 30 Trials Each, V4)

All experiments run with **identical configuration** across all 6 domains:
- 30 independent trials per experiment
- 500 iterations per trial
- 8 learners per population
- Same random seeds for reproducibility
- V4 (EG) affinity update with rate rescaling η_V4(R) = η_V3·(R²−R+1)/(R−1)

| Domain | Data Source | Records | Regimes | SI (Niche) | SI (Homo) | Cohen's d | p-value |
|--------|-------------|---------|---------|------------|-----------|-----------|---------|
| **Crypto** | Bybit Exchange | 8,766 | 4 | **0.991±0.02** | 0.002 | 58.29 | <0.001*** |
| **Commodities** | FRED (US Gov) | 5,630 | 4 | **0.990±0.02** | 0.002 | 70.48 | <0.001*** |
| **Weather** | Open-Meteo | 9,105 | 4 | **0.991±0.02** | 0.002 | 60.91 | <0.001*** |
| **Solar** | Open-Meteo | 116,834 | 4 | **0.995±0.01** | 0.002 | 100.93 | <0.001*** |
| **Traffic** | NYC TLC | 2,879 | 6 | **0.995±0.02** | 0.003 | 95.56 | <0.001*** |
| **Air Quality** | Open-Meteo | 2,880 | 4 | **0.987±0.03** | 0.002 | 54.35 | <0.001*** |
| **AVERAGE** | — | 145,294 | — | **0.992** | 0.002 | **73.4** | ✅ All |

**Key Findings (V4):**
- All 6 domains converge to SI ≥ 0.98 with statistically significant specialization (p < 0.001)
- Mean SI = 0.992; mean Cohen's d = 73.4 (every domain ≥ 54)
- Std across seeds halved relative to V3 (no clamp-driven drag)
- Traffic (R = 6) is no longer the lowest-SI domain — under V4 it reaches 0.995 on par with R = 4 domains

### Lambda Ablation Study (All 6 Domains, 30 Trials Each, V4)

| λ | Crypto | Commodities | Weather | Solar | Traffic | Air Quality | Avg |
|---|--------|-------------|---------|-------|---------|-------------|-----|
| 0.0 | 0.613 | 0.588 | 0.614 | 0.499 | 0.739 | **0.844** | 0.650 |
| 0.1 | 0.887 | 0.862 | 0.915 | 0.841 | 0.903 | 0.980 | 0.898 |
| 0.2 | 0.979 | 0.983 | 0.982 | 0.976 | 0.984 | **0.999** | 0.984 |
| **0.3** | **0.991** | **0.990** | **0.991** | **0.995** | **0.995** | 0.987 | **0.992** |
| 0.4 | 0.995 | 0.983 | 0.988 | 0.992 | 0.996 | 0.964 | 0.986 |
| 0.5 | 0.956 | 0.952 | 0.968 | 0.970 | 0.981 | 0.879 | 0.951 |

**Key Finding (V4):** Even with λ = 0 (no niche bonus), competition alone induces mean SI = 0.650 across all domains, with every domain exceeding SI = 0.49 — confirming our core thesis that **competition is sufficient for emergent specialization**. Peak performance occurs at λ ∈ [0.2, 0.4], with λ = 0.5 showing mild over-specialization in Air Quality.

### Task Performance Metrics (Illustrative)

> ⚠️ The illustrative metrics below come from `experiments/exp_task_performance.py`, which is a synthetic Monte-Carlo with hardcoded per-domain base rates and does **not** exercise the GAUSE algorithm. They are retained here only as a rough visualization. The honest task-level performance numbers are in the Method Specialization table below (which **does** run the real algorithm).

### Method Specialization Experiment (V4)

Learners choose among **5 prediction methods per domain** and specialize through competition. The per-regime method-preference update uses the V4 EG (multiplicative + renormalize) rule:

| Domain | Methods | MSI | Coverage | Niche Perf | Homo Perf | Δ% | p-value |
|--------|---------|-----|----------|------------|-----------|-----|---------|
| **Crypto** | 5 | 0.388 | 79% | 0.883 | 0.626 | **+41.2%** | <0.001*** |
| **Commodities** | 5 | 0.393 | 75% | 0.886 | 0.648 | **+36.7%** | <0.001*** |
| **Weather** | 5 | 0.426 | 99% | 0.863 | 0.675 | **+27.9%** | <0.001*** |
| **Solar** | 5 | 0.375 | 93% | 0.919 | 0.786 | **+16.9%** | <0.001*** |
| **Traffic** | 5 | 0.331 | 99% | 0.915 | 0.740 | **+23.6%** | <0.001*** |
| **Air Quality** | 5 | 0.384 | 69% | 0.912 | 0.834 | **+9.3%** | <0.001*** |
| **Average** | 5 | **0.383** | **86%** | — | — | **+25.9%** | ✅ All |

**Key Findings (V4):**
1. **Emergent Method Specialization:** Learners develop preferences for specific prediction methods (MSI = 0.383)
2. **Division of Labor:** Population uses 86% of available methods on average
3. **Performance Benefit:** Diverse populations outperform homogeneous by **+25.9%** on average
4. **Robust to update-rule choice:** V4 numbers are within ±2% of V3-era numbers on every metric, confirming that method specialization is not an artifact of the affinity-update implementation.

### MARL Head-to-Head (V4, 4 Domains, 5000 Episodes × 10 Trials)

Direct comparison against IQL, VDN, QMIX, MAPPO under V4. All methods use 8 learners and identical state/action spaces.

| Method | Crypto | Commodities | Weather | Traffic |
|---|---|---|---|---|
| **GAUSE (Ours)** | **1.000** | **1.000** | **1.000** | **1.000** |
| IQL  | 0.008 | 0.007 | 0.016 | 0.011 |
| VDN  | 0.009 | 0.007 | 0.015 | 0.011 |
| QMIX | 0.009 | 0.007 | 0.014 | 0.011 |
| MAPPO | 0.000 | 0.000 | 0.000 | 0.000 |

**Key Finding:** GAUSE reaches the maximum SI (= 1.000) in every domain while every MARL baseline stays at ≤ 0.02 — a **≥ 100× qualitative gap**. On rare-regime task rewards, GAUSE also beats the closest MARL method (IQL) by +5.1% to +8.3% per regime (+6.7% averaged).

### Method Distribution Examples

**Crypto Domain:**
- mean_revert: 45.8% of learners
- momentum_long: 38.3% of learners
- trend: 10.8% of learners
- momentum_short: 4.6% of learners
- naive: 0.4% of learners

**Traffic Domain (best diversity):**
- rush_hour: 33.8% of learners
- exp_smooth: 20.4% of learners
- weekly_pattern: 19.2% of learners
- hourly_avg: 13.8% of learners
- persistence: 12.9% of learners

---

## 📐 Prediction Methods (Mathematical Formulas)

Each domain has 5 prediction methods. Learners learn which method works best for each regime through Thompson sampling.

### 📈 Crypto Domain

| Method | Description | Formula |
|--------|-------------|---------|
| **naive** | Persistence | p̂ₜ = pₜ₋₁ |
| **momentum_short** | 5-period momentum | p̂ₜ = pₜ₋₁ + 0.1 × (pₜ₋₁ - pₜ₋₅) |
| **momentum_long** | 20-period momentum | p̂ₜ = pₜ₋₁ + 0.05 × (pₜ₋₁ - pₜ₋₂₀) |
| **mean_revert** | Mean reversion to MA20 | p̂ₜ = pₜ₋₁ + 0.2 × (MA₂₀ - pₜ₋₁) |
| **trend** | Linear trend extrapolation | p̂ₜ = pₜ₋₁ + slope(pₜ₋₁₀:ₜ) |

### 📊 Commodities Domain

| Method | Description | Formula |
|--------|-------------|---------|
| **naive** | Persistence | p̂ₜ = pₜ₋₁ |
| **ma5** | 5-day moving average | p̂ₜ = (1/5) × Σᵢ₌₁⁵ pₜ₋ᵢ |
| **ma20** | 20-day moving average | p̂ₜ = (1/20) × Σᵢ₌₁²⁰ pₜ₋ᵢ |
| **mean_revert** | Mean reversion (α=0.3) | p̂ₜ = pₜ₋₁ + 0.3 × (MA₂₀ - pₜ₋₁) |
| **trend** | 5-day trend extrapolation | p̂ₜ = pₜ₋₁ + (pₜ₋₁ - pₜ₋₅)/5 |

### 🌤️ Weather Domain

| Method | Description | Formula |
|--------|-------------|---------|
| **naive** | Persistence | T̂ₜ = Tₜ₋₁ |
| **ma3** | 3-day moving average | T̂ₜ = (1/3) × Σᵢ₌₁³ Tₜ₋ᵢ |
| **ma7** | 7-day moving average | T̂ₜ = (1/7) × Σᵢ₌₁⁷ Tₜ₋ᵢ |
| **seasonal** | Same day last week | T̂ₜ = Tₜ₋₇ |
| **trend** | 3-day trend extrapolation | T̂ₜ = Tₜ₋₁ + (Tₜ₋₁ - Tₜ₋₃)/3 |

### ☀️ Solar Domain

| Method | Description | Formula |
|--------|-------------|---------|
| **naive** | Persistence | Ĝₜ = Gₜ₋₁ |
| **ma6** | 6-hour moving average | Ĝₜ = (1/6) × Σᵢ₌₁⁶ Gₜ₋ᵢ |
| **clear_sky** | Clear sky model | Ĝₜ = G_clear(t) (theoretical max) |
| **seasonal** | Same hour yesterday | Ĝₜ = Gₜ₋₂₄ |
| **hybrid** | Weighted blend | Ĝₜ = 0.6 × Gₜ₋₁ + 0.4 × G_clear(t) |

### 🚕 Traffic Domain

| Method | Description | Formula |
|--------|-------------|---------|
| **persistence** | Last value | v̂ₜ = vₜ₋₁ |
| **hourly_average** | Historical hourly mean | v̂ₜ = v̄_h(t) where h(t) = hour of day |
| **weekly_pattern** | Same hour last week | v̂ₜ = vₜ₋₁₆₈ (168 = 24×7 hours) |
| **rush_hour_model** | Regime-based prediction | v̂ₜ = v̄_regime(t) |
| **exponential_smoothing** | EMA (α=0.3) | v̂ₜ = 0.3·vₜ₋₁ + 0.7·v̂ₜ₋₁ |

### 🌬️ Air Quality Domain

| Method | Description | Formula |
|--------|-------------|---------|
| **persistence** | Last value | q̂ₜ = qₜ₋₁ |
| **hourly_average** | Historical hourly mean | q̂ₜ = q̄_h(t) |
| **moving_average** | 24-hour MA | q̂ₜ = (1/24) × Σᵢ₌₁²⁴ qₜ₋ᵢ |
| **regime_average** | AQI regime-based | q̂ₜ = q̄_regime(qₜ₋₁) |
| **exponential_smoothing** | EMA (α=0.3) | q̂ₜ = 0.3·qₜ₋₁ + 0.7·q̂ₜ₋₁ |

### Method Categories

| Category | Methods | Best For |
|----------|---------|----------|
| **Baseline** | naive, persistence | Stable regimes, hard to beat |
| **Smoothing** | ma3, ma5, ma7, ma20, moving_average | Noisy data, reduces variance |
| **Momentum** | momentum_short, momentum_long, trend | Trending regimes |
| **Mean Reversion** | mean_revert | Volatile regimes, overshoots |
| **Seasonal** | seasonal, weekly_pattern, hourly_average | Predictable patterns |
| **Adaptive** | exponential_smoothing, hybrid | Balance between recent and history |

---

### Experimental Rigor Checklist

| Requirement | Status |
|-------------|--------|
| Same trials across all domains | ✅ 30 trials |
| Same iterations per trial | ✅ 500 iterations |
| Same number of learners | ✅ 8 learners |
| Same methods per domain | ✅ 5 methods |
| Lambda ablation on ALL domains | ✅ 6 λ values × 6 domains |
| Method specialization on ALL domains | ✅ 8 learners × 5 methods × 6 domains |
| Statistical tests on ALL domains | ✅ t-test, Cohen's d, p-value |
| Random baseline on ALL domains | ✅ 30 trials each |
| Homogeneous baseline on ALL domains | ✅ 30 trials each |
| 100% Real data | ✅ All 6 domains |

### Data Source Verification

| Domain | Source | Verification |
|--------|--------|--------------|
| 📈 Crypto | Bybit Exchange | ✅ Real exchange data with funding rates, OI, basis |
| 📊 Commodities | fred.stlouisfed.org | ✅ US Government official data (captured -$36.98 oil on 2020-04-20) |
| 🌤️ Weather | Open-Meteo API | ✅ ERA5 reanalysis + weather stations |
| ☀️ Solar | Open-Meteo Solar | ✅ CAMS satellite-derived irradiance |

---

## 🏗️ Architecture

```
GAUSE/
├── 📁 src/                           # Core implementation
│   ├── agents/                       # ⭐ Core algorithm (GAUSE)
│   │   └── niche_population.py       # NicheAgent + NichePopulation class
│   │                                 #   (implements GAUSE; V4 EG default, V3 legacy)
│   ├── domains/                      # Multi-domain data adapters
│   │   ├── crypto.py / commodities.py
│   │   ├── weather.py / solar.py
│   │   └── traffic.py / air_quality.py
│   ├── baselines/                    # Comparison baselines (IQL, VDN, QMIX, MAPPO)
│   ├── analysis/                     # SI, regret, diagnostic helpers
│   └── theory/                       # Formal propositions (Python form)
├── 📁 experiments/                   # Reproducible experiments
│   ├── _affinity_update.py           # ⭐ Shared V3/V4 update helper
│   ├── exp_unified_pipeline.py       # ⭐ Main 6-domain pipeline (V4)
│   ├── exp_capacity_division.py      # ⭐ Coverage under bounded capacity (5 arms + overlap sweep)
│   ├── exp_nonstationary_capacity.py # ⭐ Retention / catastrophic forgetting (5 arms; --soft model)
│   ├── exp_function_approx_cl.py     # ⭐ Architecture-agnostic forgetting (gradient-trained MLP experts)
│   ├── exp_hybrid_router.py          # Reward-driven router + reservation term
│   ├── exp_intra_regime_drift.py     # Stale-specialist drift + staleness trigger
│   ├── exp_oracle_fixed.py           # Oracle fixed-assignment skyline
│   ├── exp_latent_regime.py          # Class-incremental (label-free) GAUSE
│   ├── exp_population_sizing.py      # Off-diagonal N≠R coverage/retention sweep
│   ├── download_gas_data.py          # ⭐ Fetch UCI Gas Sensor Array Drift (v4.3)
│   ├── exp_real_data_gas.py          # ⭐ Real-data retention + label-free + CL baselines (v4.3)
│   ├── exp_robustness_sweep.py       # Robustness sweep (--data for real gas) (v4.3)
│   ├── exp_split_cifar_cl.py         # ⭐ Split-CIFAR-100 CNN experts (v4.3)
│   ├── plot_real_data_gas.py         # Render fig_real_data_gas.pdf
│   ├── exp_method_specialization.py  # Method specialization (V4)
│   ├── exp_marl_comparison.py        # MARL head-to-head (V4)
│   ├── exp_lambda_ablation.py        # λ ablation (V4)
│   ├── exp_lambda_zero_real.py       # λ = 0 emergence on real data (V4)
│   ├── exp_v4_v3_comparison.py       # V3 vs V4 diagnostic ablation
│   └── exp_task_performance.py       # (synthetic / illustrative)
├── 📁 tests/                         # Unit tests
│   └── test_eg_update.py             # 19 tests for V4 EG properties
├── 📁 data/                          # Datasets (committed: 6 domains; downloaded: gas, cifar)
│   ├── bybit/         commodities/   weather/
│   ├── solar/         traffic/       air_quality/
│   ├── gas_sensor/                   # UCI Gas Sensor Array Drift (download_gas_data.py; gitignored)
│   └── cifar/                        # CIFAR-100 (auto-downloaded by exp_split_cifar_cl.py; gitignored)
├── 📁 results/                       # Experiment outputs
│   ├── unified_pipeline/             # Main pipeline outputs (V4)
│   ├── capacity_division/            # Coverage results (results.json + overlap sweep)
│   ├── nonstationary_capacity/       # Retention results (+ oracle-fixed, latent-regime JSON)
│   ├── function_approx_cl/           # Gradient-trained MLP experts (permuted-digits)
│   ├── split_cifar_cl/               # ⭐ Split-CIFAR-100 CNN experts (v4.3)
│   ├── real_data/                    # ⭐ Real gas retention + robustness JSONs (v4.3)
│   ├── hybrid_router/                # Reward-driven router + reservation term
│   ├── intra_regime_drift/           # Drift + staleness trigger
│   ├── population_sizing/            # Off-diagonal N≠R sweep
│   ├── v4_v3_comparison_matched_rate/  # V3 vs V4 ablation
│   ├── real_marl_comparison/         # MARL head-to-head outputs
│   └── method_specialization/        # Method specialization outputs
├── 📁 paper/                         # LaTeX paper sources (build: latexmk -pdf in paper/)
│   ├── main.tex                      # Canonical paper (42 pages, v4.3)
│   ├── method_deep_dive.tex          # Deep-dive companion (93 pages)
│   ├── gause_explainer.tex           # System explainer (29 pages)
│   ├── figures/                      # All paper figures (committed PDFs)
│   └── references.bib
├── 📁 docs/                          # Reports + research docs
│   ├── V4_FINAL_REPORT.md            # Comprehensive V4 renovation report
│   ├── V4_EG_RENOVATION_AUDIT.md     # V3 defect audit + V4 derivation
│   ├── AUDIT_REPORT.md               # Repo-wide audit report
│   └── ARXIV_SUBMISSION_GUIDE.md     # arXiv packaging instructions
└── 📁 scripts/                       # Data download + plotting utilities
    ├── download_real_*.py            # Data downloaders
    ├── plot_v4_v3_comparison.py      # V4 vs V3 plots
    └── generate_neurips_figures.py
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/HowardLiYH/GAUSE.git
cd GAUSE

# Create conda environment
conda create -n emergent python=3.10
conda activate emergent

# Install dependencies
pip install -e .
```

### Download Real Data

```bash
# Weather (Open-Meteo - no API key needed)
python scripts/download_real_weather.py

# Solar (Open-Meteo - no API key needed)
python scripts/download_real_solar.py

# Commodities (FRED - no API key needed)
python scripts/download_fred_commodities_real.py
```

### Run Experiments

```bash
# Main 6-domain pipeline (Table 1 in the paper, V4)
python experiments/exp_unified_pipeline.py

# Headline: coverage under bounded capacity (5 arms + method-overlap sweep, fig6)
python experiments/exp_capacity_division.py

# Headline: retention under non-stationarity (5 arms, fig7; --soft adds the
# interference-model robustness check, fig8)
python experiments/exp_nonstationary_capacity.py --soft

# Closing the design space (v4.1):
python experiments/exp_function_approx_cl.py   # architecture-agnostic forgetting (needs torch)
python experiments/exp_hybrid_router.py        # router + reservation term
python experiments/exp_intra_regime_drift.py   # stale specialists + staleness trigger
python experiments/exp_oracle_fixed.py         # oracle fixed-assignment skyline
python experiments/exp_latent_regime.py        # class-incremental (label-free) GAUSE
python experiments/exp_population_sizing.py    # off-diagonal N≠R sizing

# Real-data validation (v4.3) — data is downloaded on demand, not committed:
python experiments/download_gas_data.py                          # -> data/gas_sensor/batch1..10.dat (UCI, ~5s)
python experiments/exp_real_data_gas.py --data data/gas_sensor   # retention + label-free + CL baselines on real gas
python experiments/exp_robustness_sweep.py --data data/gas_sensor # robustness sweep on the real stream
python experiments/plot_real_data_gas.py                         # -> paper/figures/fig_real_data_gas.pdf
python experiments/exp_split_cifar_cl.py                          # Split-CIFAR-100 CNN experts (needs torch+torchvision; auto-downloads CIFAR)

# Method specialization (Table 2 in the paper, V4)
python experiments/exp_method_specialization.py

# MARL head-to-head (Table 3 in the paper, V4)
python experiments/exp_marl_comparison.py

# Lambda ablation (V4)
python experiments/exp_lambda_ablation.py

# V3 vs V4 diagnostic ablation (clamp invocations, mass drift, etc.)
python experiments/exp_v4_v3_comparison.py --matched-rate

# Generate publication figures
python scripts/generate_neurips_figures.py
```

### Unit Tests

```bash
python -m pytest tests/test_eg_update.py -v
# 19/19 passing: simplex preservation, interior preservation,
# no-clamp invariance, V3/V4 first-order step-size ratio.
```

---

## 📈 SI-Performance Correlation (V3-era numbers; V4 re-derivation pending)

> The correlation analysis below was computed under V3. Because V4 collapses
> the SI distribution close to 1.0 in nearly every trial, a direct re-run of
> the same Pearson correlation under V4 is dominated by ceiling effects and is
> less informative. The qualitative conclusion (higher SI → better task
> performance) is preserved; a more diagnostic V4 version using λ-swept SI
> (where SI varies in [0.5, 1.0]) is on the v4.1 roadmap.

| Metric | Value (V3) | Interpretation |
|--------|------------|----------------|
| **Pearson r** | 0.525 | Moderate-strong positive correlation |
| **p-value** | < 0.0001 | Highly significant |
| **Regression** | Δ% = 52.9 × SI − 14.2 | Higher SI → Better performance |
| **R²** | 0.276 | SI explains 28% of performance variance |

**Per-Domain Correlation (V3):**

| Domain | r | p-value | Interpretation |
|--------|---|---------|----------------|
| Crypto | +0.411 | 0.024* | Moderate |
| Commodities | +0.591 | 0.0006*** | Strong |
| Weather | +0.349 | 0.059 | Boundary condition (P3) |
| Solar | +0.515 | 0.004** | Strong |

**Note on Weather under V3:** Weather was reported in v1.0–v3.x as a Proposition-3 boundary condition (mono-regime collapse) with the lowest SI. Under V4, Weather reaches SI = 0.991 (matching the other R = 4 domains), so the "boundary condition" framing applies to the V3 implementation rather than the underlying competitive-specialization mechanism.

---

## 🔬 Theoretical Foundation (Formal Proofs)

### Core Propositions

**Proposition 1: Competitive Exclusion** (Game-Theoretic Proof)
> In a winner-take-all game with n learners competing across k regimes, complete competitors cannot coexist at Nash equilibrium.

*Proof:* When identical strategies yield payoff V/n − c, deviation to empty niche yields V − c > V/n − c for n ≥ 2. No symmetric Nash equilibrium exists.

**Proposition 2: SI Lower Bound** (Optimization Proof)
> For niche bonus λ > 0 and k regimes: E[SI] ≥ λ/(1+λ) · (1 − 1/k)

*Proof:* Using Lagrangian optimization on the learner's reward function with entropy constraint. For λ = 0.3, k = 4: SI ≥ 0.173. Our V4 observed SI (≈ 0.99) exceeds this bound by a large margin (the bound is conservative).

**Proposition 3: Mono-Regime Collapse** (Limit Analysis)
> As dominant regime fraction η → 1, meaningful SI → 0.

*Proof:* k_eff = exp(H(regime_dist)). As η → 1, k_eff → 1, leaving nothing to specialize between.

### Additional V4-era propositions (deep-dive companion)

The full mathematical treatment is in [`paper/method_deep_dive.tex`](paper/method_deep_dive.tex) (72 pages, compiled `method_deep_dive.pdf`):

- **Prop 9.1–9.3** — Structural defects of the V3 additive heuristic (mass drift, eventual negativity, state-dependent effective rate).
- **Prop 9.4–9.6** — V4 EG update preserves the simplex by construction, preserves the interior strictly, and reduces to replicator dynamics in the small-η limit.
- **Theorem 9.1** — Hedge regret bound: the V4 update inherits the canonical $O(\sqrt{T \log R})$ regret guarantee via the Arora–Hazan–Kale potential-function argument.

---

## 📊 Figures

Five publication-quality figures in `results/figures/`:

1. **fig1_cross_domain_si.pdf** - Cross-domain SI comparison
2. **fig2_marl_comparison.pdf** - MARL baseline comparison
3. **fig3_improvement_scatter.pdf** - SI vs improvement correlation
4. **fig4_regime_distribution.pdf** - Regime distributions by domain
5. **fig5_summary_heatmap.pdf** - Summary heatmap

---

## 📋 Changelog

### v4.3.0 (2026-06-11) — Real Data: The Honest Split ⭐⭐⭐

**Major Update: synthetic-only validation replaced with real data; the contribution is reframed as a mechanism/lens, not a performance win**

- ✅ **Real retention test on UCI Gas Sensor Array Drift** (13,910 samples, 128-d, 6 classes, 10 batches / 36 months; `exp_real_data_gas.py --data`). The honest three-way split:
  1. **CL forgetting — refuted for linear-on-features, relocated to the neural regime.** A full-capacity online linear classifier does *not* catastrophically forget on real gas features (naive/replay 0.68 post-react, 0.93 overall; EWC *worse* at 0.44 — its anchor fights drift). The dissociation lives in representation-sharing neural experts.
  2. **Label-free recovery — collapses under real overlap.** GAUSE-LF post-react 0.00 (SI 0.53, cov 0.83); the label-free monolith is *also* 0.00, so the dissociation vanishes. The synthetic 1:1 method↔regime signature was load-bearing → class-incremental claim bounded to *separable-signal* regimes.
  3. **Framing.** On real data the contribution is a *mechanism and lens*, not a performance win.
- ✅ **Split-CIFAR-100 with CNN experts** (`exp_split_cifar_cl.py`, MPS/CUDA/CPU): the neural router-forgetting dissociation reproduces on a standard benchmark — GAUSE 0.581 vs router 0.748 post-react at E=R (+22%, p~10⁻³); honestly attenuated vs permuted-digits (+62%).
- ✅ **Robustness sweep on real drift** (`--data` flag): EWC degrades monotonically with its anchor (0.68→0.44→diverges); GAUSE has no knob.
- ✅ **Papers + figures updated**, abstracts/limitations aligned to the real outcome; reviewer math fixes (reallocation tail-bound factor *K*; soft-model decay exponent). All three PDFs recompiled (explainer 29pp, main 42pp, deep-dive 93pp).

### v4.2.0 (2026-06-10) — Oracle Skyline, Class-Incremental Variant, Real-Data Scaffolding ⭐⭐

- ✅ **Oracle fixed-assignment skyline** (`exp_oracle_fixed.py`): GAUSE recovers the hand-assigned partition without being given it (within 0.030 at K=1, indistinguishable for K≥3).
- ✅ **Class-incremental (label-free) variant** (`exp_latent_regime.py`): the retention result survives dropping the regime label on synthetic streams (SI 0.75, post-react 0.38 vs 1.07).
- ✅ **Reallocation-rate proposition** replacing the hand-wave necessity claim; scope made explicitly task-incremental in all three papers.
- ✅ **Real-data tooling scaffolded** (validated on a faithful surrogate; run for real in v4.3).

### v4.1.0 (2026-06-09) — Reward-Independence Reframe + Purpose-Built Baselines ⭐⭐⭐

**Major Update: the thesis is reframed around retention under bounded capacity, benchmarked against purpose-built baselines**

- ✅ **New headline result**: across five capacity-allocation arms, retention of dormant regimes tracks **reward-independence of assignment** (monolith and learned MoE router forget; random/EOI-diversity/competition retain). +71% post-reactivation vs. monolith (p < 10⁻³⁶); router fails at p ~ 10⁻³⁵.
- ✅ **Two new purpose-built baselines** in the capacity experiments: an EOI/CDS-style learned-diversity arm and a Mixture-of-Experts learned gating router.
- ✅ **Method-overlap sweep**: competition's edge over learned diversity grows monotonically with method exclusivity (−4.8% → +29.3%).
- ✅ **Idealized Observation** (paper): why a reward-driven router forgets — dormant regimes emit no protective reward signal.
- ✅ **Soft interference capacity model** (`--soft`): the dissociation survives removing LRU eviction entirely (not an artifact of discrete eviction).
- ✅ **Catastrophic-forgetting framing** with continual-learning citations; engagement with MoE-CL theory (ICLR'25, arXiv:2406.16437) — gate-freezing ⇔ reward-independent assignment.
- ✅ **Paper restructure**: new title (*Reward-Independent Capacity Assignment as a Defense Against Catastrophic Forgetting*); coverage + retention promoted to Main Results; 95% CI error bars on figs 6–8; honest-claim softening in the intro.
- ✅ **New explainer document**: `paper/gause_explainer.pdf` (17 pp) — full-system walkthrough of architecture (with diagrams), mechanisms, the reward-independence principle, experiments, and potential applications.
- ✅ **Six design-space experiments** added (`exp_function_approx_cl.py`, `exp_hybrid_router.py`, `exp_intra_regime_drift.py`, `exp_oracle_fixed.py`, `exp_latent_regime.py`, `exp_population_sizing.py`): the router's forgetting is **architecture-agnostic** (gradient-trained MLP experts, +62% at E=R); a **reservation term** recovers retention only with spare capacity (reward-independence of *protection* is the operative property); **intra-regime drift** quantifies stale specialists and a staleness-trigger remedy; an **oracle fixed-assignment skyline** shows GAUSE recovers the hand-assigned partition without being given it (within 0.030 at K=1); a **class-incremental (label-free)** variant retains (0.38 vs 1.07) with the regime label removed; and a **population-sizing** sweep yields the rule *provision N ≳ R*.

### v4.0.0 (2026-06-04) — Exponentiated-Gradient Canonical Renovation ⭐⭐⭐

**Major Update: replace the V3 additive + clamp heuristic with the canonical Hedge / multiplicative-weights update**

- ✅ **Algorithm**: niche affinity update is now the canonical exponentiated-gradient (EG) update on the regime simplex. Preserves the simplex by construction, no clamp needed, $O(\sqrt{T \log R})$ Hedge regret bound.
- ✅ **Theory**: full derivation, structural proofs of V3's mass-drift / negativity / state-dependent-rate defects, Hedge regret-bound derivation, and small-η replicator-dynamics limit (`paper/method_deep_dive.tex`, 72 pages).
- ✅ **Headline numbers strengthened (V3 → V4)**:
  - Mean SI: 0.747 → **0.992**
  - Mean Cohen's d vs. homogeneous: ≈23 → **≈73**
  - Mean SI at λ = 0: 0.329 → **0.650**
  - GAUSE vs. MARL SI gap: 4.3× → **≥100×** (1.000 vs. ≤ 0.02)
  - Traffic (R = 6): 0.573 (lowest) → **0.995** (no longer outlier)
- ✅ **Tests**: 19/19 passing in `tests/test_eg_update.py`.
- ✅ **All experiments converted to V4**; V3 retained behind `update_rule="v3_additive"` for ablation/comparison.
- ✅ **Reports**: `docs/V4_FINAL_REPORT.md`, `docs/V4_EG_RENOVATION_AUDIT.md`.
- ✅ **Release**: tagged [`v4.0.0`](https://github.com/HowardLiYH/GAUSE/releases/tag/v4.0.0) with `main.pdf` and `method_deep_dive.pdf` attached.

### v3.0.0 (2026-01-16) - Learner Populations Reframing ⭐

**Major Update: Reframed from "Multi-Agent Systems" to "Learner Populations"**

- ✅ **Terminology Update**: "agents" → "learners" throughout
- ✅ **Paper Title**: "Emergent Specialization in Learner Populations"
- ✅ **Clearer Positioning**: Distinguishes from LLM-based agents
- ✅ **arXiv Ready**: Updated paper ready for submission

### v2.0.0 (2024-12-23) - Real Data Validation

**Major Update: All experiments now use 100% verified real data**

- ✅ **4 Real Data Domains**: Crypto, Commodities, Weather, Solar
- ✅ **175K+ real records** across all domains
- ✅ **MARL Comparison**: GAUSE beats IQL by 2-4x
- ✅ **5 Publication Figures** generated
- ✅ **3 Theoretical Propositions** with proof sketches
- ✅ **Limitations Section** for honest assessment

### v1.7.0 (2024-12-22) - Unified Prediction & Mechanistic Analysis
- 📊 Unified prediction experiment across domains
- 🔬 Mechanistic analysis: why specialization works
- ⚡ Computational benchmarks: 2-4× faster than MARL

### v1.6.0 (2024-12-22) - Multi-Domain Validation
- 🚕 NYC Taxi (Traffic): SI = 0.73
- ⚡ EIA Energy: SI = 0.88
- 📈 Bybit Finance: SI = 0.86

---

## 🔬 Reproducibility

| Setting | Value |
|---------|-------|
| Random Seeds | 0-29 (30 trials per experiment) |
| Statistical Tests | Bonferroni-corrected (α = 0.05/k) |
| Confidence Intervals | 95% Bootstrap CI |
| Effect Sizes | Cohen's d reported |

**All data sources are free and publicly accessible without API keys.**

---

## 📚 Citation

```bibtex
@misc{li2026gause,
  title     = {{GAUSE}: Emergent Specialization in Learner Populations ---
               Reward-Independent Capacity Assignment as a Defense
               Against Catastrophic Forgetting},
  author    = {Li, Yuhao},
  year      = {2026},
  howpublished = {\url{https://github.com/HowardLiYH/GAUSE}},
  note      = {arXiv preprint}
}
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

<div align="center">

**⭐ Star this repo if you find it useful!**

[Report Bug](https://github.com/HowardLiYH/GAUSE/issues) • [Request Feature](https://github.com/HowardLiYH/GAUSE/issues)

</div>
