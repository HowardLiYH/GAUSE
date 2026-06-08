# Repository Restructure Plan (Post-v4.0.0)

## Purpose

This document proposes a safe, staged cleanup of the repository after the
v4.0.0 algorithm renovation. It is a **planning note only**: no code should be
moved or renamed until an explicit migration branch is created.

Goals:

1. Improve maintainability and onboarding clarity.
2. Remove legacy path/import fragility from pre-v4 code.
3. Make paper tables/claims mechanically reproducible from clear pipelines.
4. Preserve research velocity while reducing accidental breakage risk.


## Current Pain Points

The repository is scientifically strong, but structurally mixed:

- Legacy compatibility shims and stale imports coexist with v4 code paths.
- Generated artifacts (`results/*`) and source logic are tightly interleaved.
- Some modules are historical/placeholder but still importable.
- Experiment-to-paper mapping is implied, not enforced by structure.
- Test suite includes legacy tests that no longer represent active APIs.


## Target Structure (Proposed)

```text
NichePopulation/
├── src/
│   └── nichepopulation/
│       ├── algorithms/         # EG update, optional legacy adapters
│       ├── domains/            # data loaders + regime detection
│       ├── baselines/          # MARL + classical baselines
│       ├── metrics/            # SI, effect sizes, diagnostics
│       ├── pipelines/          # reusable run helpers (not CLI entrypoints)
│       └── utils/
├── experiments/
│   ├── pipelines/              # CLI experiment entrypoints only
│   └── configs/                # yaml/json configs for each paper table
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/             # golden-number checks for paper claims
├── paper/                      # LaTeX sources + figures used in paper
├── docs/
│   ├── architecture.md
│   ├── reproducibility.md
│   ├── migration_v3_to_v4.md
│   └── REPO_RESTRUCTURE_PLAN.md
├── reports/                    # human-readable summaries
└── results/                    # generated outputs (policy below)
```


## Non-Goals

- Do **not** change algorithm semantics during restructure.
- Do **not** alter v4 headline numbers during folder/module moves.
- Do **not** combine naming cleanup + algorithm changes in same PR.


## Migration Principles

1. **Small, reviewable PRs** (one structural concern per PR).
2. **Compatibility adapters first**, removals later.
3. **Move with tests**, not blind renames.
4. **Paper invariance checks** after each significant move:
   - unit tests pass
   - smoke experiments run
   - table-generating scripts still reproduce expected numbers


## Proposed Phases

### Phase 0: Freeze + Baseline (No Moves Yet)

Deliverables:

- Pin baseline commit/tag to compare against (`v4.0.0` already exists).
- Save baseline checks:
  - `pytest` pass/fail signature
  - smoke script matrix
  - headline JSON outputs for paper tables

Acceptance:

- Baseline report captured in `docs/migration_baseline.md`.


### Phase 1: Documentation and Mapping Cleanup

Deliverables:

- Add a single source-of-truth map:
  - `paper/main.tex` Table 1 -> script/config/output file
  - Table 2 -> script/config/output file
  - Table 3 -> script/config/output file
- Add `docs/architecture.md` with module boundaries and ownership.
- Add `docs/results_policy.md` (what to commit vs regenerate).

Acceptance:

- A new contributor can answer “where does this table number come from?”
  in under 2 minutes.


### Phase 2: Test Suite Modernization

Deliverables:

- Split tests into:
  - `tests/unit` (pure functions, update rules, metrics)
  - `tests/integration` (small end-to-end runs)
  - `tests/regression` (golden v4 headline checks)
- Retire/replace legacy tests currently skipped or tied to removed APIs.

Acceptance:

- `pytest` runs without legacy skips that mask broken API imports.
- Regression tests verify v4 canonical invariants.


### Phase 3: Import Hygiene and Adapter Layer

Deliverables:

- Create a single compatibility layer (if needed), e.g.
  `src/nichepopulation/compat/`.
- Route legacy imports through explicit adapters with deprecation warnings.
- Remove scattered ad-hoc shims where possible.

Acceptance:

- Import scan has zero unexpected failures.
- No module imports missing files at import time.


### Phase 4: Package Re-rooting (`src/nichepopulation/*`)

Deliverables:

- Move implementation modules under `src/nichepopulation/`.
- Keep temporary forwarding imports from old paths.
- Update `pyproject.toml` package metadata and repo URLs to current project.

Acceptance:

- Existing scripts continue working through adapters.
- New imports use `nichepopulation.*` paths.


### Phase 5: Experiments as Pipelines + Configs

Deliverables:

- Keep experiment scripts thin (`experiments/pipelines/*.py`).
- Move static knobs to config files (`experiments/configs/*.yaml`).
- Standardize output schema and folders per pipeline.

Acceptance:

- Re-running any paper table requires only one command + one config.


### Phase 6: Artifact Policy Enforcement

Deliverables:

- Decide what `results/` content is committed:
  - keep small canonical summary JSON/MD
  - avoid huge trajectory dumps by default
- Update `.gitignore` accordingly.

Acceptance:

- Git diffs remain reviewable after experiments.


### Phase 7: Adapter Removal (Final Cleanup)

Deliverables:

- Remove deprecated import paths after one full release cycle.
- Remove obsolete placeholders and dead modules.

Acceptance:

- No deprecated warnings in normal runs.
- Codebase tree aligns with documented architecture.


## Suggested PR Breakdown

1. `chore/docs-mapping-and-architecture`
2. `test/split-modernize-suite`
3. `chore/import-compat-layer`
4. `refactor/package-reroot-src-nichepopulation`
5. `refactor/experiment-pipelines-and-configs`
6. `chore/results-artifact-policy`
7. `chore/remove-legacy-adapters`


## Risk Register

1. **Silent numeric drift** after refactor  
   Mitigation: regression tests on canonical outputs.

2. **Broken downstream scripts due to path changes**  
   Mitigation: temporary forwarding modules + deprecation warnings.

3. **Huge PRs hard to review**  
   Mitigation: phase-based PR boundaries above.

4. **Mixing cleanup with scientific changes**  
   Mitigation: freeze algorithm logic during restructure branch.


## Definition of Done (Restructure)

- Clear package boundaries and import paths.
- `pytest` suite reflects current APIs and passes cleanly.
- Paper table reproduction is explicit and scripted.
- Results artifact policy documented and enforced.
- Legacy adapters removed (or clearly quarantined with timeline).


## Recommended Timeline

- Phase 0-2: 2-4 days
- Phase 3-5: 4-7 days
- Phase 6-7: 2-3 days

Total: approximately 1.5-2 weeks part-time, or 4-6 focused days.


## Immediate Next Step (When Ready)

Create a dedicated branch (example):

```bash
git checkout -b chore/repo-restructure-v4-followup
```

Start with **Phase 1 only** (docs + mapping), open PR, and merge before any
module moves.
