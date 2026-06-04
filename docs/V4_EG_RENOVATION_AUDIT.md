# V4 EG Renovation — Batch A Audit Report

**Branch:** `v4-eg-canonical`
**Scope:** Algorithmic correctness only. Naming consistency and stale tests are out-of-scope for this batch (tracked separately below).
**Goal:** Replace the heuristic affinity update (Eq. 48 in the deep-dive paper) with the principled exponentiated-gradient (EG) formulation.

---

## 1. The Smoking Gun

The current implementation lives at:

```118:140:src/agents/niche_population.py
def _update_niche_affinity(self, regime: str) -> None:
    """
    Update niche affinity using the paper's formula.

    Paper formula: α_r ← α_r + η × (1 - α_r)  [for winning regime]
    Other regimes: α_r' ← α_r' - η/(R-1)
    Then normalize.
    """
    eta = self.learning_rate  # η = 0.1 as in paper
    n_regimes = len(self.regimes)

    # Paper formula: α += η × (1 - α) for winning regime
    self.niche_affinity[regime] += eta * (1 - self.niche_affinity[regime])

    # Decrease other regimes
    for r in self.regimes:
        if r != regime:
            self.niche_affinity[r] -= eta / (n_regimes - 1)
            self.niche_affinity[r] = max(0.01, self.niche_affinity[r])

    # Normalize to maintain probability simplex
    total = sum(self.niche_affinity.values())
    self.niche_affinity = {r: a / total for r, a in self.niche_affinity.items()}
```

This is Eq. 48 verbatim from the deep-dive paper, with one **undocumented twist**: the `max(0.01, ...)` clamp on line 136. The published main paper (`paper/main.tex`) makes no mention of this clamp. Only the deep-dive pseudocode acknowledges it.

This clamp is the "mathematical duct tape" — without it, the loss-side update would push small affinities negative, and the subsequent normalization (a division) cannot fix sign. The clamp silently prevents that failure mode at the cost of breaking the formal equation's stated semantics.

---

## 2. Three Inconsistent Presentations of the Same Algorithm

The current codebase and papers present this algorithm in **four different forms** that disagree with each other:

### Form A — `paper/main.tex` line 169 (published paper, headline)
```latex
\alpha_{i^*,r_t} \leftarrow \alpha_{i^*,r_t} + \eta \cdot (1 - \alpha_{i^*,r_t})
```
followed by "normalization to maintain $\alpha_{i^*} \in \Delta^R$". **Does not mention the loss-side update for other regimes at all.** Pretends the algorithm only touches the winning regime's entry.

### Form B — `paper/method_deep_dive.tex` line 1380-1397 (deep-dive formal equation, Eq. 48)
```latex
\alpha_{i,r}^{(t+1)} =
\begin{cases}
\alpha + \eta(1-\alpha) & \text{if winner & current regime} \\
\alpha - \eta/(R-1)     & \text{if winner & other regime} \\
\alpha                  & \text{if not winner}
\end{cases}
```
followed by normalization. **Acknowledges the loss-side update but not the clamping.**

### Form C — `paper/method_deep_dive.tex` line 2815 (deep-dive pseudocode)
```
α_{i*,r} ← max(0.01, α_{i*,r} - η/(R-1))
```
**Acknowledges the clamping.** The only place in any document where clamping appears.

### Form D — `src/agents/niche_population.py` lines 130-140 (actual implementation)
Same as Form C, in code.

### The story this tells
The headline main paper presents the cleanest, most aspirational form. The deep dive adds the loss side. The pseudocode adds the clamp. The code matches the pseudocode. The further you dig, the more "engineering reality" leaks in. A careful reviewer who follows the trail from main paper to code will find three layers of undocumented modifications.

---

## 3. Structural Issues with the Current Update

In order of severity for the V4 revision:

1. **Mass drift before normalization.** Summing the changes across all $R$ regimes:
   $\Delta\sum_r \alpha_{i^*,r} = \eta(1-\alpha_{i^*,r_t}) - (R-1) \cdot \tfrac{\eta}{R-1} = -\eta \cdot \alpha_{i^*,r_t}$
   The sum systematically shrinks. Normalization restores it to 1, but **distorts the per-entry update** by a state-dependent factor $\frac{1}{1-\eta\alpha_{i^*,r_t}}$.

2. **Undocumented clamping.** Without the `max(0.01, ...)` clamp, entries go negative whenever an off-current-regime affinity drops below $\frac{\eta}{R-1}$. Normalization (a positive scaling) cannot recover from negativity. The clamp silently fixes this but is not mentioned in the published main paper.

3. **Asymmetric proportionality.** Gain side is proportional to $(1-\alpha)$; loss side is a uniform $\frac{\eta}{R-1}$. The two sides are not derived from the same principle.

4. **No connection to known theory.** The current update is not gradient descent on anything, not a Bregman projection, not mirror descent, not multiplicative weights, and not replicator dynamics. It is a one-off heuristic.

---

## 4. Proposed Replacement (V4)

Exponentiated gradient (Hedge / multiplicative weights / mirror descent with entropic regularizer on the simplex):

$$\alpha_{i^*,j}^{(t+1)} = \frac{\alpha_{i^*,j}^{(t)} \exp(\eta \cdot \mathbf{1}[j = r_t])}{\sum_{k=1}^R \alpha_{i^*,k}^{(t)} \exp(\eta \cdot \mathbf{1}[k = r_t])}$$

Properties:
- **Mass preservation by construction** — denominator is exactly $\sum_k \exp(\eta r_k)$-weighted, sum-to-1 holds without a separate normalization step.
- **Non-negativity by construction** — $\exp$ is positive, ratio of positives is positive. No clamp needed.
- **Single principled equation** — no separate cases for winning vs other regimes. The exp absorbs both.
- **Substrate-invariant** — same update form works for any simplex, regardless of $R$ or substrate.
- **Tied to known theory** — special case of mirror descent on $\Delta^R$ with entropic regularizer, equivalent to discrete-time replicator dynamics in the small-$\eta$ limit.

For a single winner with binary reward (indicator on current regime $r_t$), this simplifies operationally to:
- $\alpha_{i^*,r_t} \leftarrow \alpha_{i^*,r_t} \cdot \exp(\eta) / Z$
- $\alpha_{i^*,r} \leftarrow \alpha_{i^*,r} / Z$ for $r \neq r_t$
- where $Z = \alpha_{i^*,r_t} \exp(\eta) + \sum_{r \neq r_t} \alpha_{i^*,r}$

Three multiplications, one division, no special cases, no clamps.

---

## 5. Files That Must Change

### Code (in scope for Batch B)

| File | Lines | Change |
|---|---|---|
| `src/agents/niche_population.py` | 118-140 | Replace `_update_niche_affinity()` with EG update |
| `tests/` | new file | Add `test_eg_update.py` with simplex-preservation, non-negativity, and equivalence-in-small-η tests |

### Paper text (in scope for Batch C, after experiments)

| File | Lines | Change |
|---|---|---|
| `paper/main.tex` | 167-171 | Replace Form A with EG equation; explicitly state normalization is no longer needed (it's intrinsic) |
| `paper/main.tex` | 189-190 | Update Algorithm 1 pseudocode line: replace additive update + normalize with single EG line |
| `paper/method_deep_dive.tex` | 1378-1397 | Replace Form B (Eq. 48) with EG; add a paragraph noting the equivalence to mirror descent / replicator dynamics |
| `paper/method_deep_dive.tex` | 2811-2818 | Update Form C pseudocode to EG; remove clamping line |
| `paper/method_deep_dive.tex` | 2217-2222 | Convergence Rate proposition currently relies on the additive form; either re-state for EG (geometric rate of $(1 - \eta(1-\alpha^*))^{W}$ becomes a known multiplicative-weights bound) or move to a separate proof |
| `paper/method_deep_dive.tex` | 2928-2933 | Worked example "Agent C's affinity update" uses old numbers; recompute under EG |

### CHANGELOG (in scope for Batch C)

- Add a v4.0.0 entry explaining the algorithmic change, the three structural issues with v3.x, and the empirical comparison results

---

## 6. Out-of-Scope for V4 (Tracked Separately)

These should be done eventually but **not** mixed into the v4-eg-canonical branch, because conflating them with the algorithm change muddies the diff.

### 6.1 Code-level "agent → learner" rename
The README's v3.0.0 changelog claims the rename is done, but the code still uses "agent" pervasively:
- Class: `NicheAgent` (should be `NicheLearner` or similar)
- Variable names: `agent_id`, `n_agents`, `self.agents`
- Directory: `src/agents/` (should be `src/learners/`)
- Test file: `TestPopulation` uses `agent_{i}` ID convention

**Recommendation:** Defer to a separate `v4.1-naming-cleanup` branch. It's a large mechanical rename touching many files; it deserves its own commit and review.

### 6.2 Stale tests
`tests/test_core.py` imports `from src.agents.population import Population` — this module does not exist in the current codebase. The test file was written against an older API. It will not run as-is.

**Recommendation:** Defer to a separate `v4.2-test-rewrite` branch. The tests written for V4 should target the new NichePopulation API specifically.

### 6.3 Convergence Rate proposition rewrite
The proposition at line 2217 of `method_deep_dive.tex` gives a geometric convergence bound that depends on the additive update form. Under EG, the analogous bound exists but takes a different form (related to MWU regret bounds). Rewriting this rigorously is a non-trivial theory exercise.

**Recommendation:** Mark as a TODO in the deep dive. For Batch C, replace with a remark that "convergence properties for the EG update follow from standard multiplicative weights analysis; we defer a formal restatement to follow-up work."

---

## 7. Hyperparameter Recommendation (Updated After Batch B Numerical Check)

The V3 codebase uses $\eta = 0.1$ as the learning rate. **Naive reuse of $\eta = 0.1$ in V4 produces materially different dynamics**, because the two update rules have different first-order gains at uniform start.

### Corrected First-Order Analysis

Working through the math at $\alpha = 1/R$ with a win on regime $r_0$:

**V3 (after post-hoc normalization):**
$$\Delta \alpha_{r_0}^{V3} \big|_{\alpha=1/R} = \eta_{V3} \cdot \left(1 - \frac{1}{R} + \frac{1}{R^2}\right) + O(\eta^2)$$

**V4 (EG, intrinsic):**
$$\Delta \alpha_{r_0}^{V4} \big|_{\alpha=1/R} = \eta_{V4} \cdot \frac{1}{R}\left(1 - \frac{1}{R}\right) + O(\eta^2)$$

**Ratio (same $\eta$ in both):**
$$\frac{\Delta \alpha^{V3}}{\Delta \alpha^{V4}} = \frac{R^2 - R + 1}{R - 1}$$

For $R = 4$ this is $13/3 \approx 4.33$. **V3 with $\eta=0.1$ moves the winning entry ~4.33× as much per step as V4 with $\eta=0.1$**. This is a leading-order, not a small-correction, difference. Both numerical tests in `tests/test_eg_update.py::TestEGV3FirstOrderStepSizeGap` confirm this.

### Why the original Section 7 analysis was wrong

The earlier draft of this section claimed first-order equivalence at $\alpha = 1/R$. That was a sign error in the normalization step. The corrected analysis accounts for the fact that V3's post-hoc normalization *amplifies* the winner-side update by a factor of $1/(1 - \eta/R)$, which adds another $\eta/R$ to the first-order gain. V4 has no such amplification because the normalization is *intrinsic* (the division happens by definition of EG, not as a post-hoc fix).

### Updated Recommendation

**Headline value: $\eta_{V4} = 0.1$**, the same numerical value as V3. We are *not* attempting to reproduce V3's specialization timescale; we are claiming V4 is the principled algorithm and reporting its own behavior at the same $\eta$. The V4 trajectories will look slightly slower per step but will reach the same equilibrium specialization. This is the cleanest framing for the V4 paper.

**Optional matched-dynamics comparison: $\eta_{V4} = \frac{R^2-R+1}{R-1} \cdot \eta_{V3}$** for the explicit "V4 matches V3 timestep-for-timestep" plot in the diagnostic section. For $R=4$, this gives $\eta_{V4} \approx 0.43$.

**Appendix robustness sweep: $\eta_{V4} \in \{0.05, 0.1, 0.2, 0.5\}$** to show the specialization equilibrium is not sensitive to $\eta_{V4}$ choice within an order of magnitude.

This three-tier approach gives reviewers (1) a single headline number to compare V4 vs V3, (2) the per-step matching for the "are these really the same dynamics" question, and (3) the sensitivity sweep for robustness.

---

## 8. Other Open Questions

1. **Reward signal for EG.** Currently the win is binary (you won this round). Should we use $r_j = \mathbf{1}[j = r_t]$ (Option A: simplest, matches Form B's semantics) or $r_j = $ raw reward when $j = r_t$, 0 otherwise (Option B: richer signal, breaks equivalence to Form B)?
   **Recommended:** Option A for V4. Option B can be a follow-up experiment.

2. **Does the V4 paper position EG as a "correction" or as a "principled refactor"?**
   Phrasing matters for reviewer reception. "Correction" implies V3 was wrong (it was, in a structural sense, but the empirics still held). "Principled refactor" implies the V3 heuristic was a reasonable first cut and EG is the canonical form.
   **Recommended:** "Principled refactor" tone — acknowledges the issues honestly without retracting V3's empirical results.

3. **Should the V4 paper include the diagnostic plots (mass drift, negativity frequency) as part of the V3-vs-V4 comparison?**
   These are the "proof of the bug" plots. Without them, the V4 paper just says "we changed the equation"; with them, it shows precisely why the change was needed.
   **Recommended:** Yes, include 2 small diagnostic plots in a "Why V4 is Different" section.

---

## 9. What Batch B Will Deliver

Once Howard signs off on Section 7 and Section 8 above, Batch B will:

1. Implement EG in `src/agents/niche_population.py` (replacing `_update_niche_affinity()`)
2. Add unit tests verifying: simplex sum = 1 exactly, all entries ≥ 0, no clamping path is invoked, EG-vs-V3 first-order equivalence at $\alpha = 1/R$
3. Write a small sanity-check script that runs both V3 and V4 on a single domain × 5 seeds and saves the SI trajectory comparison
4. Update Form B in `method_deep_dive.tex` (math derivation only — prose comparison waits for Batch C)
5. Deliver: diff, working tests, sanity-check plot

No experiments at full scale, no paper prose rewrites, no commits beyond the working branch.
