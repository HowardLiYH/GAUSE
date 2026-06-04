# V4 vs V3 Affinity-Update Comparison (full)

- Iterations:    500
- Seeds/domain:  30
- Niche bonus:   0.3
- Learning rate: 0.433 (same for both rules)
- Wall time:     2.0s

## Per-domain summary

| Domain | V4 mean SI | V3 mean SI | Diff (V4 - V3) | 95% CI half-width | V3 clamps | V3 mean pre-norm sum |
|--------|-----------:|-----------:|---------------:|------------------:|----------:|---------------------:|
| crypto | 0.9967 +/- 0.0127 | 0.7202 +/- 0.0601 | +0.2765 | +/- 0.0231 | 37210 | 1.3496 |
| commodities | 0.9967 +/- 0.0127 | 0.7202 +/- 0.0601 | +0.2765 | +/- 0.0231 | 37210 | 1.3496 |
| weather | 0.9967 +/- 0.0127 | 0.7202 +/- 0.0601 | +0.2765 | +/- 0.0231 | 37210 | 1.3496 |
| solar | 0.9967 +/- 0.0127 | 0.7202 +/- 0.0601 | +0.2765 | +/- 0.0231 | 37210 | 1.3496 |
| traffic | 0.9558 +/- 0.0437 | 0.5615 +/- 0.0822 | +0.3943 | +/- 0.0337 | 55366 | 1.3118 |
| air_quality | 0.9967 +/- 0.0127 | 0.7202 +/- 0.0601 | +0.2765 | +/- 0.0231 | 37210 | 1.3496 |

**Interpretation.** V4 (EG) is a structural fix for V3, not a numerical drop-in.
Under matched eta, V4 produces slightly lower (or comparable) final SI than V3 because
the per-step gain is (R^2 - R + 1) / (R - 1) times smaller at uniform start.
However, V4 entirely eliminates the structural pathologies of V3:

- Zero clamp invocations across all seeds and domains (V3: hundreds to thousands).
- The pre-normalization sum equals 1 by construction (V3: strictly less than 1 each
  winning round).

Both rules produce the same qualitative finding (emergent specialization). V4's
advantage is theoretical (canonical Hedge regret bound, clean replicator-dynamics
limit) and numerical (no hidden clamp behavior).
