# Real-Data Validation & Robustness: Experiment Plan

**Purpose.** Close the two standing weaknesses of the GAUSE evaluation:

1. **Synthetic-data reliance.** The headline retention result and the
   class-incremental (label-free) result were validated only on synthetic streams
   where regime `r` is signalled 1:1 by method `r`. Real validation is needed on a
   stream with genuine recurring regimes, real drift, and *no* method↔regime
   bijection.
2. **Strongest numbers vs weakest baselines.** The +71% / +62% figures are versus a
   capacity-bounded monolith and a vanilla router. Against the honest competitors
   (learned diversity, hybrid-with-reservation, oracle skyline) the result is
   parity. We need (a) a comparison to *real continual-learning* baselines, and
   (b) a way to convert "parity" into a defensible claim.

The plan below is the minimum set of runs that fixes both. Two scripts are already
written and validated on a faithful offline surrogate (`exp_real_data_gas.py`,
`exp_robustness_sweep.py`); they run on the real data unchanged.

---

## 1. Datasets (all free), matched to GAUSE's three preconditions

GAUSE needs: (i) recurring regimes, (ii) bounded per-learner capacity, (iii)
dormancy + reactivation. Ranked by fit and by how directly they attack the two
weaknesses.

| Dataset | Regimes | Why it fits | Attacks weakness | Source (free) |
|---|---|---|---|---|
| **UCI Gas Sensor Array Drift** | 6 gas classes | real 128-d sensor features, real drift over 36 months, 10 batches; **no method↔regime bijection** | #1 (synthetic), #2 (real CL) | `archive.ics.uci.edu/dataset/224` |
| **UCI HAR / PAMAP2 / WISDM** | human activities | activities recur & go dormant; real accelerometer features; naturally class-incremental | #1 | UCI ML Repo |
| **CWRU / NASA IMS bearings** | fault modes | rare recurring faults = the paper's "rare-event retention" story, real signals | #1 | CWRU / NASA PCoE |
| **UCI Electricity Load / ENTSO-E** | seasonal/daily modes | strong recurring regimes, clean dormancy | #1 | UCI / ENTSO-E |
| **CORe50** | 50 objects, 11 sessions | purpose-built class-incremental with recurrence; lets us cite CL SOTA | #1, #2 | `vlomonaco.github.io/core50` |
| **CLEAR** | natural temporal shift | real imagery, smooth real drift over a decade | #1 | `clear-benchmark.github.io` |
| **Ken French / FRED (multi-decade)** | market regimes | crisis regimes genuinely dormant for years (single-cycle crypto is the current gap) | #1 | FRED, Ken French Data Library |

**Primary recommendation:** Gas Sensor Array Drift (cheapest path to a real
retention + real label-free result) and one CL benchmark (CORe50) to put GAUSE on
the same axis as continual-learning baselines.

---

## 2. Experiments

### E1 — Real-data retention (attacks #1)
Stream the gas batches in time order with a sliding active-window dormancy schedule
on top of the real drift. Arms: GAUSE-LF, capacity-bounded monolith, oracle-fixed
(skyline), learned-diversity. **Metric:** post-reactivation accuracy.
**Prediction:** GAUSE retains; the bounded monolith forgets dormant gas classes
across the long inter-occurrence gaps. *Script:* `exp_real_data_gas.py --data PATH`.

### E2 — Class-incremental label-free recovery (attacks #1, the deepest one)
Same stream, **no class label at decision time.** Each agent specializes over the
real feature space; winner = nearest prototype = implicit class estimate. **Metric:**
post-reactivation accuracy + specialization/coverage of the recovered partition vs
the hidden classes. **Prediction:** GAUSE recovers most of the oracle skyline
*without labels*, with imperfect (honest) coverage. This is the run that proves the
method↔regime proxy was not load-bearing.

### E3 — Real continual-learning baselines (attacks #2)
Add supervised CL competitors on the same stream: online softmax with **naive SGD**,
**EWC**, **experience replay**. These are the baselines reviewers asked for. **Metric:**
post-reactivation accuracy. **Prediction (real data):** naive forgets; EWC/replay
partially retain at their tuned best; GAUSE-LF (no labels, no buffer, no Fisher)
lands in the same band — the legitimate "parsimony" claim, now against real CL.

### E4 — Hyperparameter-robustness sweep (converts #2 parity → robustness)
Sweep each competitor's knob (learned-div repulsion β, EWC λ, replay buffer) over a
wide range; GAUSE is a flat reference (no knob). **Deliverable:** the *shape* — the
competitor matches GAUSE only in a narrow knob band and degrades when mis-tuned;
GAUSE lands there for free. *Script:* `exp_robustness_sweep.py`.

### E5 — CL-benchmark comparison (optional, for CL-community framing)
Run the function-approximation version on CORe50 (or Split-CIFAR) with N experts as
capacity, against EWC/replay/ER. Puts GAUSE on a standard axis and answers "why not
just use a CL method?" directly.

---

## 3. Prototype results already obtained (offline surrogate)

A structurally faithful surrogate (128-d, overlapping, drifting Gaussian classes,
**no 1:1 signature**) validates the code and the label-free mechanism:

- **Label-free GAUSE: post-reactivation accuracy 0.875, SI 0.86, coverage 1.00** —
  recovers all six hidden classes with no labels and retains across dormancy,
  reaching **88% of the oracle skyline (1.00) without labels**.
- Capacity-bounded label-free monolith: **0.319** (forgets) → GAUSE **+55.6 pts**.
- Robustness sweep: learned-diversity runs from 0.875 (β=0, i.e. plain competition)
  down to **0.0** when mis-tuned (spread 0.875); **GAUSE is flat at 0.875 with no
  knob.** This is the quantified parsimony-as-robustness claim.
- Honest caveat: on this benign surrogate the supervised CL baselines (EWC/replay)
  sit at 1.00 — there is nothing to forget — so the E3/E4 *CL* curves are
  uninformative until run on the real severe-drift data. That is exactly the run to
  do.

**What the surrogate already shows:** the label-free recovery does **not** depend on
the synthetic 1:1 proxy — it works on high-dim overlapping drifting features. That
is the single most important thing the reviewers doubted.

---

## 4. Success criteria (be honest about what would weaken the paper)

- **Strong outcome:** on real gas data, GAUSE-LF retains while the monolith/naive-CL
  forget, GAUSE-LF ≈ tuned EWC/replay (parity, no machinery), and the robustness
  sweep shows competitors degrade off-optimum. → Both weaknesses answered.
- **Honest-null outcome:** if real overlap is so severe that label-free coverage
  collapses (SI low, coverage ≪ 1), report it — it bounds the class-incremental
  claim to "separable-signal regimes," which is already the stated scope. This would
  *weaken* the label-free headline but strengthen credibility.
- **Watch item:** if EWC/replay *beat* GAUSE on real data by a clear margin, the
  contribution narrows to "parsimony/interpretability," and the paper should be
  framed as a mechanism/lens paper, not a performance paper.

---

## 5. How to run

```bash
# 1. Get the data (free): UCI Gas Sensor Array Drift -> unzip batch1.dat..batch10.dat
#    https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset

# 2. Real retention + label-free + CL baselines (E1–E3)
python experiments/exp_real_data_gas.py --data path/to/gas_batches/

# 3. Robustness sweep (E4)
python experiments/exp_robustness_sweep.py            # uses same arms

# Offline sanity (no network) — runs the faithful surrogate:
python experiments/exp_real_data_gas.py
```

Outputs: `results/real_data/gas_real.json`, `robustness_sweep.json`, and figures.
