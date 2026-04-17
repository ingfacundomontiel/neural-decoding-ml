# Neural Decoding ML — Restructure Plan

## Goal

Rebuild `neural-decoding-ml/` into a clean, reproducible, thesis-grade workspace for
neural decoding research. Primary supported workflows:

- **(b) Reproduce past experiments** deterministically (regenerate figures, numbers)
- **(c) Compare results** across datasets, regions, and animal cohorts
- **(d) Audit and extend `ibioml`** — find bugs, fix them, contribute fixes upstream
- **(e) Deep exploration** of raw datasets (the first task)

This project is **the thesis codebase** for a master's degree. Every decision must
be traceable in writing. Documentation is not optional.

## Non-goals (for now)

- No new experimental runs. No new model architectures. No hyperparameter sweeps.
- No production CLI tooling, no packaging as a pip library, no CI/CD.
- No migration of past experiment *results* from `marinburginlab`. Only raw datasets.
- No effort to preserve compatibility with the legacy `neural-decoding-ml` code —
  it moves to `_legacy/` and is treated as read-only reference.

## Current situation (one line each)

- `neural-decoding/` — exploratory notebooks, April 2025, messy root.
- `neural-decoding-ml/` — stalled OOP rewrite, August 2025, known bugs documented.
- `marinburginlab-labPrograms/neural-decoding/` — production experiments using `ibioml`,
  October/November 2025, flat scripts, includes a monkey-patch for an ibioml bug.

Full findings in `research.md`.

## Key decisions

Rationale for each in `decisions.md`. Summary:

| Decision | Choice |
|---|---|
| Repo location | Rebuild inside `neural-decoding-ml/`, git history kept |
| Legacy handling | Move current contents to `_legacy/`, read-only |
| `ibioml` strategy | Fork to `facundomontiel/IBioML`, git submodule at `external/ibioml/` |
| Package manager | `uv` + `pyproject.toml` + `uv.lock` |
| Python version | 3.11 (pinned in `.python-version`) |
| Notebook format | Jupytext-paired (`.ipynb` + `.py` percent) |
| Docs | Plain markdown in `docs/` and `specs/` |
| Language | English everywhere |

## Target layout

```
neural-decoding-ml/
├── specs/                      # plans, research, decisions (this directory)
│   ├── plan.md
│   ├── research.md
│   ├── decisions.md
│   └── dataset-inventory.md
├── data/
│   ├── raw/                    # deduplicated .mat files — one copy of each
│   │   └── README.md           # dataset catalog (region, animal, session, bins)
│   └── processed/              # reproducible pickles, gitignored by default
├── external/
│   └── ibioml/                 # git submodule → facundomontiel/IBioML fork
├── exploration/                # jupytext-paired notebooks + outputs
├── analysis/                   # Python modules supporting exploration (plotting, stats)
├── experiments/                # reproducible configs — empty for now
├── docs/                       # thesis-grade documentation
│   └── ibioml-audit.md         # written during phase 1
├── tests/                      # unit tests for analysis/ and (later) ibioml fixes
├── _legacy/                    # archived previous neural-decoding-ml contents
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
├── .gitmodules
└── README.md
```

## Phased roadmap

### Phase 0 — Bootstrap (one sitting)

1. Archive current `neural-decoding-ml/*` to `_legacy/`.
2. Scaffold new directory tree (empty dirs with `.gitkeep` or placeholder READMEs).
3. Fork `mariburginlab-labPrograms/IBioML` → `facundomontiel/IBioML` via `gh`.
4. Add the fork as submodule at `external/ibioml/`.
5. Initialize `pyproject.toml` with `uv`, Python 3.11, core deps
   (numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, torch, jupytext, jupyterlab).
6. Set up `.gitignore` (venv, `__pycache__`, `.DS_Store`, `data/processed/`, `_legacy/` stays tracked for now).
7. Commit: `chore: bootstrap new project structure`.

**Deliverable:** fresh tree, `uv sync` works, submodule cloned, `jupyter lab` starts.

### Phase 1 — Dataset migration + audit prep

1. Inventory every `.mat` file across the three source directories
   (recorded in `dataset-inventory.md`).
2. Deduplicate: latest-authoritative wins, one copy per dataset → `data/raw/`.
3. Write `data/raw/README.md` with the full catalog: filename, region, animal,
   session, bin size, cohort (expert/naive/hippocampus), known issues.
4. Read every module of `external/ibioml/ibioml/` and fill in `docs/ibioml-audit.md`
   with: module map, function-by-function behavior, spotted bugs, code-quality
   concerns, test coverage assessment, extension ideas. Start with the
   known `preprocess_data` index-type bug.
5. Commit progress files per logical chunk.

**Deliverable:** `data/raw/` canonical, `docs/ibioml-audit.md` complete,
`specs/dataset-inventory.md` final.

### Phase 2 — Dataset exploration (the actual priority)

Build exploration notebooks under `exploration/`, each paired as `.ipynb` + `.py`:

1. `01-dataset-overview.py` — per-dataset shapes, trial counts, NaN inventory,
   variables present/missing, cross-dataset consistency table.
2. `02-behavior.py` — position trajectories, velocity distributions, trial durations,
   dPrime over trials, criterion, goResponse patterns, rewarded vs unrewarded trial balance.
3. `03-neural-firing.py` — per-neuron firing rate histograms, low-firing neurons
   by dataset, distribution across the population, temporal patterns within trials.
4. `04-context.py` — rewCtxt/rewOdor balance, trial-type breakdowns.
5. `05-cross-region-comparison.py` — piriform vs hippocampus structural differences
   (neuron counts, trial counts, firing rates, behavior).

Support code lives in `analysis/` (plotting functions, stats helpers, .mat loaders).
Iterate based on what the data shows — this list is a starting point, not a contract.

**Deliverable:** clear picture of every dataset, reusable plotting/stats code,
findings written up in `docs/dataset-findings.md`.

### Phase 3 — ibioml fixes (triggered by audit findings)

Once audit is complete, prioritize fixes/features:

1. Fix the known `preprocess_data` index-type bug (remove the monkey-patch).
2. Address other bugs found in phase 1.
3. Write tests under `external/ibioml/tests/` for each fix.
4. Open PRs upstream from the fork.

No speculative refactoring of `ibioml`. Only touch what the audit flagged.

### Phase 4 — Reproducibility of past experiments (later)

When the user is ready, use `experiments/` with a config-driven approach to
reproduce SAN/SFN/neuron-scaling results using the patched `ibioml`. Out of
scope for now.

## Workflow rules

- **Every non-trivial decision goes in `decisions.md`** with date and rationale.
- **Every significant finding goes in `research.md`** so future-you has context.
- **Commit often** with semantic commit messages.
- **Tests for any fix to `ibioml`** — even if upstream currently has none.
- **No scope creep.** If a tangent comes up, note it in `specs/backlog.md` (create
  on demand) and keep moving.

## Backlog (things intentionally deferred)

- Config-driven experiment runner
- Cross-experiment result aggregation / comparison dashboard
- Upstreaming fixes to ibioml PyPI release
- CI (linting, tests on PR)
- MkDocs site (if the lab adopts the project)
