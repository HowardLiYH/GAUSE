# Reproducibility Guide

This document provides complete instructions to reproduce all results in the paper *"Emergent Specialization in Learner Populations: Competition as the Source of Diversity"* (v4.0.0).

> **As of v4.0.0** the niche affinity update is the canonical exponentiated-gradient (Hedge / multiplicative-weights) update. To reproduce the V3-era numbers from v1.0–v3.x of the paper, see the "Legacy V3 reproduction" section at the end.

## Quick Start (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/HowardLiYH/NichePopulation.git
cd NichePopulation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run all experiments (V4 by default)
python experiments/exp_unified_pipeline.py
```

## Repository Structure

```
NichePopulation/
├── paper/                     # LaTeX source
│   ├── main.tex               # Canonical paper (26 pages, V4)
│   ├── method_deep_dive.tex   # Deep-dive companion (72 pages, V4)
│   ├── references.bib
│   └── figures/
├── src/                       # Core library
│   ├── agents/                # NichePopulation algorithm (V4 default)
│   ├── baselines/             # MARL baselines (IQL, VDN, QMIX, MAPPO)
│   ├── domains/               # 6 domain implementations
│   ├── analysis/              # SI, regret, diagnostic helpers
│   └── theory/                # Formal propositions
├── experiments/               # Reproducible experiments
│   ├── _affinity_update.py    # Shared V3/V4 update helper
│   ├── exp_unified_pipeline.py     # Main 6-domain pipeline
│   ├── exp_method_specialization.py
│   ├── exp_marl_comparison.py
│   ├── exp_lambda_ablation.py
│   ├── exp_lambda_zero_real.py
│   └── exp_v4_v3_comparison.py     # V3 vs V4 diagnostic
├── tests/                     # Unit tests
│   └── test_eg_update.py      # 19 tests for V4 EG properties
├── data/                      # Real-world data (6 domains, 145K records)
├── results/                   # Experiment outputs
├── docs/                      # Reports + research docs
│   ├── V4_FINAL_REPORT.md
│   └── V4_EG_RENOVATION_AUDIT.md
└── scripts/                   # Data download + figure generation
```

## System Requirements

- **Python**: 3.8+
- **Memory**: 4GB RAM
- **Storage**: 500MB for data
- **Time**: ~30 minutes for full experiments

## Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
```

## Experiments

### Main Experiment: Unified Pipeline

Runs all core experiments across all 6 domains:

```bash
python experiments/exp_unified_pipeline.py
```

**Output**: `results/unified_pipeline/results.json`

### Individual Experiments

| Experiment | Command | Output |
|------------|---------|--------|
| Hypothesis Tests | `python experiments/exp_hypothesis_tests.py` | `results/hypothesis_tests/` |
| Method Specialization | `python experiments/exp_method_specialization.py` | `results/method_specialization/` |
| MARL Head-to-Head | `python experiments/exp_marl_comparison.py` | `results/real_marl_comparison/` |
| λ Ablation | `python experiments/exp_lambda_ablation.py` | `results/lambda_ablation/` |
| V3 vs V4 Diagnostic | `python experiments/exp_v4_v3_comparison.py --matched-rate` | `results/v4_v3_comparison_matched_rate/` |

### Unit Tests

```bash
python -m pytest tests/test_eg_update.py -v
# 19/19 passing: simplex preservation, interior preservation,
# no-clamp invariance, V3 mass-drift, V3/V4 first-order step-size ratio.
```

### Generate Figures

```bash
python scripts/generate_neurips_figures.py
python scripts/plot_v4_v3_comparison.py  # V3 vs V4 trajectories + diagnostics
```

## Data

All data is **100% real-world data** from verified public sources:

| Domain | Source | Records |
|--------|--------|---------|
| Crypto | Bybit Exchange | 44,000+ |
| Commodities | FRED (US Gov) | 5,630 |
| Weather | Open-Meteo | 9,105 |
| Solar | Open-Meteo | 116,834 |
| Traffic | NYC TLC | 2,879 |
| Air Quality | Open-Meteo | 2,880 |

See `data/README.md` for download instructions.

## Expected Results (v4.0.0)

Running the unified pipeline under the default V4 (EG, rescaled-η) configuration should produce:

| Domain | SI (Niche) | SI (Homo) | Cohen's d |
|--------|------------|-----------|-----------|
| Crypto | 0.991 ± 0.024 | 0.002 | 58.29 |
| Commodities | 0.990 ± 0.020 | 0.002 | 70.48 |
| Weather | 0.991 ± 0.023 | 0.002 | 60.91 |
| Solar | 0.995 ± 0.014 | 0.002 | 100.93 |
| Traffic | 0.995 ± 0.015 | 0.003 | 95.56 |
| Air Quality | 0.987 ± 0.026 | 0.002 | 54.35 |
| **Mean** | **0.992** | **0.002** | **73.4** |

All differences are statistically significant at p < 0.001 (Welch's t-test). Numbers are reproducible with seed = 42 across 30 trials per condition.

### Legacy V3 reproduction

To reproduce v1.0–v3.x numbers (mean SI ≈ 0.747, Cohen's d ≈ 23):

```bash
# Force the V3 additive heuristic in the unified pipeline
python experiments/exp_unified_pipeline.py --update-rule v3_additive

# Or via the V3 vs V4 ablation
python experiments/exp_v4_v3_comparison.py --matched-eta
```

See `docs/V4_FINAL_REPORT.md` for the side-by-side comparison and `docs/V4_EG_RENOVATION_AUDIT.md` for the rationale for the V3 → V4 transition.

## Random Seeds

All experiments use fixed random seeds for reproducibility:
- Base seed: 42
- Per-trial seed: 42 + trial_index

## Statistical Methods

- **Trials per experiment**: 30
- **Statistical test**: Welch's t-test (two-sided)
- **Significance threshold**: α = 0.001
- **Effect size**: Cohen's d
- **Multiple testing**: Bonferroni correction where applicable

## Docker

```bash
docker build -t emergent-specialization .
docker run emergent-specialization python experiments/exp_unified_pipeline.py
```

## Contact

For questions about reproducibility:
- **Author**: Yuhao Li
- **Email**: li88@sas.upenn.edu
- **Repository**: https://github.com/HowardLiYH/NichePopulation
- **v4.0.0 Release**: https://github.com/HowardLiYH/NichePopulation/releases/tag/v4.0.0
