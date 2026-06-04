# Experiments

This folder contains the key experiments for the paper *"Emergent Specialization in Learner Populations: Competition as the Source of Diversity"* (v4.0.0).

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
- `results/v4_v3_comparison_matched_rate/` — V3 vs V4 diagnostic plots + trajectories
