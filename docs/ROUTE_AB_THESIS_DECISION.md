# Route A / Route B: Thesis Framing Decision

Status: investigation complete. **Outcome: the STRONG claim is rescued on real
data** (traffic / commodities / solar) once the right domains + the coupled
architecture are used. Route A holds as a complementary result.
Date: 2026-06-06
Owner experiments:
- `experiments/exp_route_b_diagnostic.py` (oracle structure gate)
- `experiments/exp_route_b_coupled.py` (coupled architecture + synthetic vs real test)
- `experiments/exp_domain_screen.py` (domain screener + real-data validation)

> **TL;DR (post domain-hunt + authentication):** the original 3 domains
> (energy/weather/finance) genuinely lack regime structure (oracle 0-1.9%) but
> are unrepresentative. Screening the wider catalogue + a leakage audit leaves
> two clean domains with **clock-based (causal) regimes** where real beats the
> shuffled control by ~+13-15% (traffic, solar). Copper's apparent +13.7% was a
> regime-label **leakage artifact** and is discarded. See Sections 7-8.
>
> **CRITICAL CORRECTION (Section 9, ablations):** that real-vs-shuffled gap is a
> **regime-CONDITIONING** effect, NOT a population-SPECIALIZATION effect.
> (a) Turning the niche->winner coupling on/off makes no difference (paired
> p=0.36). (b) A *single* regime-conditioned agent matches or beats the
> specialized population (population is -26% WORSE on solar). So the **blanket
> claim ("emergent specialization improves performance") is NOT supported**;
> what is supported is that regimes carry exploitable info and a conditioned
> learner uses it. The defensible thesis is **Route A**: robust coordination-free
> division of labor that *tracks* exploitable structure, not a performance lever.
>
> **POSITIVE RESULTS (Sections 10-11):** the ablations used an
> *unlimited-capacity, stationary* monolith. Two realistic conditions give
> specialization a genuine accuracy edge: **(10) bounded capacity** (`K << R`,
> regime-exclusive methods) -- emergent specialization beats both a
> capacity-matched monolith and an equal-capacity random-diversity population
> (synthetic +57%, real traffic +8.7% at `K=1`); and **(11) non-stationarity**
> with finite memory -- the population acts as distributed persistent memory and
> beats a capacity-bounded monolith by +33% overall / +71% post-reactivation
> (p~1e-37) by retaining dormant-niche expertise the monolith must relearn. So
> specialization IS an accuracy lever, precisely when capacity or memory is the
> binding constraint; we claim exactly that scope.

---

## 0. The fork we were resolving

The redesigned negative control (`exp_regime_shuffle.py`) showed that high
Specialization Index (SI) does **not** translate into a task-performance benefit
on the real domains: shuffling regime labels leaves performance unchanged. That
threatens the **strong thesis**:

> *Strong thesis (environment-driven):* learners specialize **because** the
> environment rewards regime-specific expertise; the specialization is
> performance-relevant and structure-dependent.

Two responses were on the table:

- **Route B (rescue the strong claim):** the failure might be architectural --
  niche affinity is computed but never feeds back into behavior. Re-couple it
  and the performance benefit might appear.
- **Route A (fallback framing):** drop the performance claim and reframe around
  **coordination-free division of labor** -- a self-organization result that does
  not require a task-performance gap.

Per direction: *investigate Route B first (cheap, could rescue the strong claim),
prepare Route A as fallback, then proceed.* This document records the outcome.

---

## 1. Confirmed: the current pipeline decouples niche from behavior

In the real-data prediction pipeline (`exp_lambda_zero_real.PredictionPopulation`):

- `select_method(regime)` samples from **per-regime method beliefs** only.
- the winner each step is the agent with the **lowest prediction error**.
- niche affinity is updated (EG) but never enters method selection **or** winner
  determination -- it is a dead-end readout.

So niche affinity cannot affect performance in the current code *by construction*.
This is a necessary precondition for Route B to even be testable.

---

## 2. Oracle gate: is there structure to capture at all? (decisive)

`exp_route_b_diagnostic.py` ignores the algorithm and asks a property of the
**data + predictors**: if you knew the regime perfectly and always used the
regime-optimal method (SPECIALIST oracle), how much better than the single
globally-best method (GENERALIST oracle) could you do?

| Domain  | Best method per regime                         | Specialist-vs-generalist ceiling |
|---------|------------------------------------------------|----------------------------------|
| energy  | `RenewableAware` in **every** regime           | **+0.00%**                       |
| weather | `Seasonal` in **every** regime                 | **+0.00%**                       |
| finance | `Volatility` in 3/4, `MeanRevert` in `volatile`| **+1.89%**                       |

**Reading:** energy and weather have a *globally dominant* method -- the regime
never changes the optimal choice. Finance has genuine but tiny regime structure.
The oracle is an upper bound: **no algorithm, no matter how it couples niche to
behavior, can beat these numbers.** On the current domains the achievable
performance benefit of regime-specialization is essentially zero.

This localizes the problem precisely: the binding constraint is the
**(regime definition x method set)**, *not* the learning algorithm.

---

## 3. Coupled architecture: does the mechanism work when structure exists?

`exp_route_b_coupled.py` implements Route B properly:

1. **Coupled winner**: `score_i = relative_quality_i + lambda*(affinity_i[regime] - 1/R)`
   -- a niche-matched agent gets a leg up, but `relative_quality` is **real**
   prediction quality (not a random draw, unlike the headline pipeline).
2. **Winner-take-all learning**: only the winner updates its per-regime method
   belief, so a regime's specialist accumulates clean expertise there.
3. **Routing readout**: the population's prediction comes from the agent whose
   primary niche == current regime.

We then run a **real-vs-shuffled** comparison in two settings:

| Condition                          | real-vs-shuffled gap | p        | Cohen's d | verdict        |
|------------------------------------|----------------------|----------|-----------|----------------|
| **Synthetic regime-champion**      | **+68.0%**           | 8e-46    | -27.8     | mechanism works|
| Real: energy                       | -1.0%                | 0.62     | +0.10     | no structure   |
| Real: weather                      | +2.3%                | 0.004    | -0.87     | artifact*      |
| Real: finance                      | +1.8%                | 0.053    | -0.52     | at ceiling     |

\* weather's gap is **not** method-optimality (oracle = 0%); it is a second-order
effect where coherent labels build slightly cleaner per-regime beliefs. It is
within the oracle noise floor and not the strong-thesis story.

**Reading:**
- The synthetic task (method `m` is best in regime `m` by construction) yields a
  **+68% gap that vanishes under shuffling** -- the coupled mechanism captures
  regime structure decisively *when it exists*. **The algorithm is not broken.**
- The real domains show <=2% gaps, consistent with their oracle ceilings. The
  coupled algorithm roughly *saturates* finance's 1.9% ceiling. There is simply
  almost nothing to capture.

**Route B conclusion:** the architecture is validated, but it does **not** rescue
the strong claim *on the current real domains*, because those domains lack
exploitable regime-method structure. Rescuing the strong claim requires
**different data/tasks** (large oracle gap), not a different algorithm.

---

## 4. Route A (fallback): coordination-free division of labor

The same runs report a population-diversity / **niche-coverage** metric
(fraction of regimes claimed as some agent's primary niche):

| Domain  | coverage (real) | coverage (shuffled) |
|---------|-----------------|---------------------|
| synthetic | 1.00          | 0.99                |
| energy  | 1.00            | 1.00                |
| weather | 1.00            | 1.00                |
| finance | 0.88            | 0.93                |

**Reading:** the population reliably partitions into distinct niches (coverage
~= 1.0) **whether or not the labels are meaningful**. Division of labor is
therefore a robust property of the **competitive dynamics** (competitive
exclusion among agents), independent of environmental structure.

This is the honest, supported Route A claim:

> *Route A (supported):* the algorithm reliably self-organizes a **stable,
> reproducible, coordination-free division of labor** -- distinct agents occupy
> distinct niches without communication. This is a property of the population
> dynamics, demonstrated robust to label shuffling.

What Route A must **not** claim: that the division reflects or exploits
environmental structure (coverage is identical under shuffling), nor that it
improves task performance on these domains (it does not).

---

## 7. Domain hunt: the strong claim DOES hold on the right real data

`exp_domain_screen.py` sweeps the wider data catalogue with a shared generic
method set (`persistence, momentum, mean_revert, ma_short, ma_long, drift`) and
ranks each (dataset x regime-scheme) by specialist-oracle gap.

Top real candidates (oracle gap):

| dataset / scheme            | R | distinct winners | oracle gap |
|-----------------------------|---|------------------|------------|
| solar_denver / level        | 3 | 2                | 14.6%      |
| traffic_nyc / **native**    | 6 | 2                | 10.1%      |
| commodity_copper / **native**| 4 | 3               | 6.3%       |

Crypto series (BTC/ETH/SOL/DOGE) show ~0% -- one method dominates, like the
original domains. So energy/weather/finance were simply *unrepresentative*.

**Validation with the coupled algorithm (real vs shuffled, n=20 trials each):**

| domain (regimes)              | routed err real | shuffled | gap     | p       | d     |
|-------------------------------|-----------------|----------|---------|---------|-------|
| traffic_nyc (native semantic) | 723.1           | 854.5    | +15.4%  | 3e-10   | -2.59 |
| commodity_copper (native)     | 366.9           | 425.1    | +13.7%  | 5e-5    | -1.37 |
| solar_denver (level)          | 136.3           | 166.3    | +18.1%  | 3e-13   | -3.36 |

The coupled algorithm routes to regime specialists and **beats the shuffled
control by 14-18%, all p < 1e-4**. The gap exceeds the oracle gap because the
shuffled condition corrupts the per-regime belief tables (worse than the
generalist baseline), making the destruction of structure especially visible.

**Headline material:** `traffic_nyc` and `commodity_copper` use the dataset's
*own semantic regime labels* (rush hours / market phases) -- no circularity --
and still show large, highly significant gaps. These are the cleanest evidence
for the strong claim. (`solar_denver` uses a data-derived level scheme that is
partly diurnal; use it as supporting, not headline.)

This is the negative control *passing*: real regimes beat shuffled, exactly as
the strong thesis predicts. The earlier "thesis overthrown" conclusion was an
artifact of testing on three structure-free domains.

---

## 8. Data authentication (provenance + leakage)

Before headlining the Section 7 domains, we authenticated them.

### 8.1 Regime-label causality (the key risk)

A regime label is only a legitimate routing signal if it is known **before** the
target value -- otherwise routing leaks the target and inflates the
real-vs-shuffled gap. Findings:

| domain  | native regime source                                   | causal? |
|---------|--------------------------------------------------------|---------|
| traffic | hour-of-day + weekday (clock) -- `src/domains/traffic.py` | YES    |
| solar   | clear-sky index from the **current** GHI               | NO (leaks) |
| copper  | price vs current MA + **global** vol percentile         | NO (leaks) |

The screener's data-driven schemes were also made strictly causal (label for
step `i` uses only `values[:i]`, expanding/past-only percentiles -- see
`exp_domain_screen.py`). Re-running with causal labels:

| domain (causal regime)        | oracle gap | coupled gap | p      | survives? |
|-------------------------------|-----------:|------------:|--------|-----------|
| traffic_nyc (clock, native)   | 10.1%      | **+15.4%**  | 3e-10  | YES       |
| solar_denver (hour-of-day)    | 15.9%      | **+12.8%**  | 5e-9   | YES       |
| solar_denver (causal level)   | 12.4%      | +18.1%      | 4e-16  | YES (diurnal proxy) |
| commodity_copper (causal)     | <1%        | ~0          | n.s.   | **NO -- was leakage** |

**Copper's earlier 13.7% was a leakage artifact** and is discarded. Traffic and
solar survive with clock-based regimes that cannot leak by construction. Solar's
per-regime optima are physically sensible (morning->persistence,
midday->mean_revert, evening->momentum).

### 8.2 Predictor causality

All generic methods (`persistence, momentum, mean_revert, ma_short, ma_long,
drift`) use `history[:idx]` only -- no peeking. Clean.

### 8.3 Provenance

| domain  | source (README)        | statistical check                                   | verdict |
|---------|------------------------|-----------------------------------------------------|---------|
| traffic | NYC TLC, Jan-Apr 2023  | 2,879 int counts; autocorr lag1=0.94, lag24=0.90, **lag168=0.92** (real daily+weekly seasonality) | real* |
| solar   | Open-Meteo, 2023       | 116,834 rows; realistic GHI; engineered clear-sky features via `download_real_solar.py` | real    |

\* Traffic has **no download script** (README says "see NYC TLC API"), so it is
not bit-for-bit reproducible. The data is statistically consistent with real
taxi counts, but for the thesis we should either add a fetch script or treat
reproducibility as a caveat.

**Authentication verdict:** the strong claim rests on **traffic** and **solar**
with **clock-based (causal) regimes**, on data that is statistically real.
Copper is dropped. Crypto/energy/weather/finance remain structure-free controls.

---

## 9. Ablations (decisive): coupling and specialization are not the accuracy lever

`exp_route_b_validation.py` runs two controlled ablations on **9 series** with
clock-based (causal) regimes: four NYC-taxi month-windows and all five solar
locations. Each arm uses 20 trials.

### 9.1 Coupled vs decoupled (does niche->winner coupling matter?)

Identical pipeline, only the niche term in winner determination toggled.

| group   | coupled mean gap | decoupled mean gap |
|---------|------------------|--------------------|
| traffic | +14.9%           | +12.0%             |
| solar   | +9.7%            | +9.2%              |
| all 9   | **+12.0%**       | **+10.4%**         |

Paired t-test coupled > decoupled: **p = 0.36 (n.s.)**. The coupling does not
add accuracy. The real-vs-shuffled gap is produced by **per-regime method
selection** (shuffling corrupts the regime->method map), which both arms share.

### 9.2 Specialized population vs single regime-conditioned agent

The decisive test of whether *specialization* (not just conditioning) buys
accuracy: one agent with per-regime beliefs, learning from every step.

| series group | population err | single-agent err | population better by |
|--------------|----------------|------------------|----------------------|
| traffic (4)  | ~727           | ~766             | +4.9% (mixed sig.)   |
| solar (5)    | ~134           | ~107             | **-25.6% (worse)**   |

The single conditioned agent **beats** the specialized population on solar by a
wide margin (it learns each regime from all data; winner-take-all specialization
fragments the population's data). Traffic is a slight, inconsistent population
edge. Net: **specialization is not a reliable accuracy lever; it can hurt.**

### 9.3 What this means

- **Supported:** (i) some real domains have exploitable regime structure
  (oracle, real>>shuffled); (ii) a regime-conditioned learner exploits it;
  (iii) the population robustly self-organizes a division of labor (coverage~=1,
  even under shuffle) without communication.
- **Not supported:** that emergent *population specialization* (or the
  niche->winner coupling) improves task accuracy over a conditioned monolith.

The honest, defensible thesis is therefore **Route A**, upgraded with the
structure-tracking result: *the algorithm is a coordination-free
self-organization mechanism whose niches align with performance-relevant regimes
when such structure exists* -- framed as an organization/dynamics contribution,
with an explicit, well-controlled account of when specialization does and does
not help accuracy (it tracks structure; it is not an accuracy lever).

---

## 10. Capacity-bounded division of labor: a genuine (narrow) accuracy win

`experiments/exp_capacity_division.py`. The Section 9 ablations compared the
population to a monolith with **unlimited capacity** -- precisely the regime
where a generalist is feasible and division of labor is unnecessary. The
realistic condition for any finite learner is **bounded per-agent capacity**:
each agent can master only `K` of `R` regimes. We give every agent competence
in only its top-`K` niche-affinity regimes (uninformed elsewhere) and sweep `K`.

Three arms, **all with the same per-agent capacity `K`**:
- **monolith**: one capacity-`K` agent (covers at most `K` of `R` regimes);
- **random-diversity**: `N=R` agents with *fixed random* capacity-`K` niches
  (diversity WITHOUT competition -- the control for "just more total capacity");
- **specialized**: `N=R` competitive agents (emergent division of labor).

### 10.1 Synthetic regime-champion (R=6, exclusive optimal method per regime)

| K | monolith | random-div | specialized | spec vs mono | spec vs random | p(vs random) |
|---|----------|-----------|-------------|--------------|----------------|--------------|
| 1 | 0.934 | 0.577 | 0.249 | **+73%** | **+57%** | 3e-17 |
| 2 | 0.803 | 0.333 | 0.254 | **+68%** | **+24%** | 4e-5 |
| 3 | 0.659 | 0.245 | 0.255 | +61% | -4% (n.s.) | 1.0 |
| 4 | 0.520 | 0.245 | 0.259 | +50% | -6% (n.s.) | 1.0 |
| 6 | 0.245 | 0.245 | 0.259 | -6% | -6% | 1.0 |

### 10.2 Traffic (NYC, real, R=6 clock regimes)

| K | monolith | random-div | specialized | spec vs mono | spec vs random | p(vs random) |
|---|----------|-----------|-------------|--------------|----------------|--------------|
| 1 | 1083 | 929 | 848 | **+22%** | **+8.7%** | 1.3e-4 |
| 2 | 1005 | 821 | 814 | +19% | +0.9% (n.s.) | 0.32 |
| 3 | 906 | 790 | 810 | +11% | -2.6% (n.s.) | 0.93 |

Solar (R=3) shows **no** advantage (too few regimes; methods overlap across
regimes, so narrow specialists lose to broad shallow coverage).

### 10.3 What this establishes (and its precise scope)

- **vs a capacity-matched monolith**, specialization wins decisively for `K<R`
  on both synthetic and real (traffic) data -- but part of this is simply more
  total capacity (`N*K` vs `K`).
- **vs random diversity at equal total capacity**, competition/emergent
  specialization adds accuracy **only in the tight-capacity regime** (`K=1`,
  sometimes `K=2`), where random niche assignment leaves coverage gaps but
  competitive exclusion guarantees coverage. Once `K` is moderate, random
  diversity already covers the regime space and competition is unnecessary.
- This is consistent with Section 9: with slack capacity, specialization is not
  an accuracy lever. The new, honest positive result is that **emergent
  specialization IS the accuracy lever precisely when per-agent capacity is
  tight (`K << R`) AND regimes have (near-)exclusive optimal methods.** In that
  corner it beats both a monolith and random diversity; outside it, it does not.

This is a genuine, well-scoped performance claim -- not the original blanket
"specialization improves performance," but a clean characterization of *when* it
does. It strengthens the thesis without overclaiming.

---

## 11. Non-stationarity: population as distributed persistent memory

`experiments/exp_nonstationary_capacity.py`. Section 10's advantage is *spatial*
(coverage of a regime space larger than one agent's capacity). Non-stationarity
exposes a second, *temporal* mechanism: **retention**. Regimes cycle -- only a
sliding window of `W=3` of `R=6` regimes is active per epoch, so each regime goes
dormant for several epochs then **reactivates**. Each agent has a bounded belief
store of `K` regimes with **LRU eviction** (finite memory).

- **monolith** (1 agent, `K` slots): serves whatever is active now, evicting
  dormant regimes; on reactivation it must **relearn from scratch**.
- **random-diversity** / **specialized** (`N=R` agents): each owns a niche and
  idles while it is dormant, so it **retains** that niche's beliefs and is ready
  the instant the regime reactivates.

Result (R=6, W=3, 1920 steps, 20 trials), error = overall / post-reactivation:

| K | monolith | random-div | specialized |
|---|----------|-----------|-------------|
| 1 | 1.010 / 1.015 | 0.575 / 0.603 | **0.239 / 0.283** |
| 3 | 0.359 / 0.906 | 0.239 / 0.252 | **0.240 / 0.260** |
| 6 | 0.239 / 0.253 | 0.239 / 0.252 | 0.240 / 0.255 |

At `K=3 (<R)` the specialized population beats the capacity-matched monolith by
**+33% overall (p=4e-37)** and **+71% in the post-reactivation window
(p=3e-39)**: the gap is concentrated exactly where the monolith relearns
(reactivation error 0.906 vs 0.260, true reactivations only). The population is
flat at ~0.24 for *all*
`K` -- even `K=1` agents collectively form a complete persistent memory the
monolith only matches at `K=R`. (At `K=1` competition also beats random diversity,
0.239 vs 0.575, by guaranteeing coverage; they converge for `K>=2`.)

**Two independent conditions now give specialization a genuine accuracy edge:**
(i) bounded capacity vs environmental complexity (Section 10, spatial coverage);
(ii) non-stationarity with finite memory (Section 11, temporal retention). Both
are realistic for any finite learner, and both are absent from the
unlimited-capacity stationary monolith used in the Section 9 ablations -- which is
why those (correctly) found no advantage.

---

## 5. Recommendation

After the Section 9 ablations, the recommendation is the **honest Route A
framing, upgraded with structure-tracking** -- NOT a performance-superiority
claim:

1. **Lead with coordination-free division of labor.** The population reliably
   partitions niches (coverage ~= 1.0) without communication, robust to label
   shuffling. This is the core self-organization contribution.
2. **Add the structure-tracking result.** On domains with genuine, causally
   labelled regime structure (traffic, solar), the niches align with
   performance-relevant regimes; the regime signal is exploitable (real >>
   shuffled). On structure-free domains (energy/weather/finance/crypto, oracle
   ~0) nothing is claimed -- the oracle diagnostic is the suitability test.
3. **Report the ablations as honest negatives.** The niche->winner coupling adds
   no accuracy (p=0.36) and the specialized population does not beat a single
   regime-conditioned agent (worse on solar). State plainly that specialization
   here is an organizational phenomenon, not an accuracy lever.
4. **Claim the conditional accuracy result (Section 10), scoped tightly.** Under
   bounded per-agent capacity (`K << R`) with regime-exclusive methods, emergent
   specialization beats both a capacity-matched monolith and an equal-capacity
   random-diversity population (synthetic + real traffic). State the scope: this
   is the *only* regime where specialization is an accuracy lever; with slack
   capacity it is not.
5. **Do NOT** use copper (leakage), and do NOT claim an unconditional
   performance advantage.

This is credible and reviewer-proof: a real self-organization phenomenon, a
clean structure-tracking result with a passing negative control, a transparent
account of the limits, AND a well-controlled positive result that pinpoints
exactly when division of labor pays off (tight capacity). The capacity result
(Section 10) is the realized version of the previously-"future-work"
higher-upside path; the remaining future work is non-stationary regimes and
explicit per-agent cost models.

---

## 6. Revised "everything else" (was Phase 1 diagnostics)

- The "why does the same eta give different SI per domain?" question is partly
  answered: SI is driven by regime count R and label imbalance, and is *not*
  tied to exploitable structure (oracle = 0% yet SI ~= 0.9). Quantify R /
  occupancy-entropy vs SI to close it out.
- Data authentication (are the CSVs real/accurate) is now **more** important: if
  we pursue Route B, the screening must run on trustworthy data.
- Update `paper/main.tex` and `docs/V4_FINAL_REPORT.md` to match whichever
  framing is chosen.
