# V4 vs V3 Affinity-Update Comparison (smoke)

- Iterations:    200
- Seeds/domain:  5
- Niche bonus:   0.3
- Learning rate: 0.433 (same for both rules)
- Wall time:     0.0s

## Per-domain summary

| Domain | V4 mean SI | V3 mean SI | Diff (V4 - V3) | 95% CI half-width | V3 clamps | V3 mean pre-norm sum |
|--------|-----------:|-----------:|---------------:|------------------:|----------:|---------------------:|
| crypto | 0.7490 +/- 0.1086 | 0.7238 +/- 0.0647 | +0.0252 | +/- 0.1232 | 2382 | 1.3296 |

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
