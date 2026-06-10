# Experiments

This folder contains the key experiments for the paper *"Emergent Specialization in Learner Populations: Reward-Independent Capacity Assignment as a Defense Against Catastrophic Forgetting"* (v4.1.0).

> **As of v4.0.0**, all experiments use the canonical exponentiated-gradient (Hedge / multiplicative-weights) update on the niche affinity by default. The V3 additive heuristic is retained behind `update_rule="v3_additive"` for ablation and comparison.

## Quick Start: Reproduce All Results

```bash
# From repository root
python experiments/exp_unified_pipeline.py
```

This runs the main experiment across all 6 domains with identical configuration and the V4 EG affinity update.

## Experiment Files

| File | Purpose | Headline V4 Finding |
|------|---------|---------------------|
| `exp_unified_pipeline.py` | **Main reproducibility script** | Mean SI = 0.992, Cohen's d = 73.4 across 6 domains |
| `exp_capacity_division.py` | ⭐ Coverage under bounded capacity (5 arms: monolith, random, EOI-diversity, MoE router, competition; plus method-overlap sweep) | Competition wins at K=1 (+57% synthetic, +8.7% traffic); edge over learned diversity scales with method exclusivity (−4.8% → +29.3%) |
| `exp_nonstationary_capacity.py` | ⭐ Retention / catastrophic forgetting under non-stationarity (same 5 arms; `--soft` runs the interference-decay robustness model) | Retention tracks reward-independence 5-for-5: monolith + MoE router forget dormant regimes (p < 10⁻³⁶, p ~ 10⁻³⁵); random/diversity/competition retain |
| `exp_function_approx_cl.py` | ⭐ Function-approximation continual learning: gradient-trained MLP experts (permuted-digits), monolith / MoE router / GAUSE, capacity = #experts E (requires `torch`) | Router forgetting is **architecture-agnostic**: at E=R, GAUSE post-reactivation test error 0.218 vs MoE router 0.576 (+62% lower, p ~ 10⁻⁹); MoE does not improve with E |
| `exp_hybrid_router.py` | Hybrid arm: reward-driven router **+ memory-preservation (reservation) term**, vs vanilla MoE / monolith / GAUSE | Reservation recovers most retention at K≥2 (+48% at K=3, p ~ 10⁻¹¹) but **cannot help at K=1** (no spare slot); GAUSE retains even at K=1 — reward-independence of *protection* is the operative property |
| `exp_intra_regime_drift.py` | Intra-regime concept drift: champion method drifts on reactivation; standard GAUSE vs a staleness-trigger variant | Stale specialists hurt: standard GAUSE overall error 0.90 (worse than a K=3 monolith, 0.35); a cheap staleness trigger recovers most of it (0.43, ~53% lower) |
| `exp_population_sizing.py` | Off-diagonal population sizing: sweep N (agents) vs fixed R=6 at K∈{1,2}; coverage, redundancy, retention | Scarce agents (N<R) under-cover, governed by N **not** N·K (one primary niche per agent); abundant agents (N≫R) idle harmlessly (redundant = N−R) with no retention loss |
| `exp_method_specialization.py` | Method specialization analysis | Learners specialize in prediction methods (+25.9% improvement, 86% method coverage) |
| `exp_marl_comparison.py` | Head-to-head MARL comparison | NichePop SI = 1.000 vs. IQL/VDN/QMIX ≤ 0.016, MAPPO = 0.000 (≥100× gap) |
| `exp_lambda_ablation.py` | λ ablation study | λ = 0.3 is the sweet spot; mean SI ≥ 0.98 in λ ∈ [0.2, 0.4] |
| `exp_lambda_zero_real.py` | λ = 0 on real data | Mean SI = 0.65, every domain ≥ 0.39 (competition alone) |
| `exp_v4_v3_comparison.py` | V3 vs V4 diagnostic ablation | Quantifies clamp invocations, mass drift, and SI gap between V3 and V4 |
| `exp_hypothesis_tests.py` | Formal hypothesis testing (H1–H4) | All four hypotheses pass at p < 0.001 |
| `exp_mechanism_ablation.py` | Isolate competition vs. bonus effects | Competition is the primary driver |
| `exp_regime_shuffle.py` | Negative control | Specialization robust to regime relabeling |
| `exp_rare_regime_resilience.py` | Resilience to rare regimes | Specialists adapt to rare regimes |
| `_affinity_update.py` | Shared V3/V4 update helper | Used by all of the above |
| `exp_task_performance.py` | ⚠️ Synthetic / illustrative only | **Does not run the algorithm**; see docstring |

## Configuration

All experiments use identical settings (see `config.py`):

| Parameter | Value |
|-----------|-------|
| Trials | 30 |
| Iterations | 500 (5,000 in MARL comparison) |
| Learners | 8 |
| Default λ | 0.3 |
| Default update rule | `eg` (V4) |
| Rate rescaling for V4 | η_V4(R) = η_V3 · (R² − R + 1) / (R − 1) |

## Results

Results are saved to `../results/` as JSON files alongside figures. The main V4 outputs live in:

- `results/unified_pipeline/results.json` — Table 1
- `results/method_specialization/results.json` — Table 2
- `results/real_marl_comparison/results.json` — Table 3
- `results/capacity_division/results.json` — coverage under bounded capacity (fig6) + overlap sweep
- `results/nonstationary_capacity/results.json` — retention / forgetting (fig7); `results_soft.json` — soft-model robustness (fig8)
- `results/function_approx_cl/results.json` — function-approximation CL with gradient-trained experts (fig11)
- `results/hybrid_router/results.json` — hybrid reward-driven router + reservation term (fig9)
- `results/intra_regime_drift/results.json` — intra-regime concept drift + staleness trigger (fig10)
- `results/population_sizing/results.json` — off-diagonal N≠R coverage/redundancy/retention sweep (fig12)
- `results/v4_v3_comparison_matched_rate/` — V3 vs V4 diagnostic plots + trajectories
