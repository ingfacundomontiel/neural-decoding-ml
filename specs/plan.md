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

### Phase 5 — Single-target, temporal-window, and confounding-factor experiments

Follow-up experimental phase after the SFN-poster batch. The rule for this
phase is **one change at a time**: every item below modifies exactly one
factor vs. the shared baseline. A single run cannot combine two items.

**Shared baseline** (anchor for every delta):

- MLP dual output, `bins_before=2 / bins_current=1 / bins_after=2` (1000 ms window).
- 14 datasets (17 total minus `DG-Naive-S38`, `DG-Naive-S41`, `Pir-Expert-N59`,
  excluded from plots).
- `_withCtxt_flat.pickle` variants (rewCtxt as extra feature).
- Hyperparam space matching the lab reference: hidden_size 128–512 step 64,
  num_layers 1–3, dropout 0–0.5, lr log 1e-8→1e-3, batch 32, es_patience 20,
  num_epochs 200. Optuna 20 trials, 5 outer folds × 1 inner hold-out.

#### Item 1 — Single-output experiments (pos-only, vel-only) — Priority 1

Same baseline, only `output_size=1` and a different pickle target.

- **Datasets**: all **17** (no plot-time exclusions applied at run time; we
  filter at plotting stage if needed).
- **`num_exclusive_layers`**: omitted from the model_space for single-target
  (only applies when there are multiple heads).
- **Context**: kept ON — `_withCtxt_onlyPosition_flat.pickle` and
  `_withCtxt_onlyVelocity_flat.pickle`, mirroring the dual baseline.
- **Volume**: 17 × 2 = 34 experiments.
- **Estimated GPU wall-time**: ~25–35 h (single tends to be a bit faster
  than dual due to smaller output layer).
- **Results dirs**: `results/mlp_<dataset>_bins200ms_pos/` and `_vel/`,
  emitted by `scripts/run_all_experiments.py --targets pos,vel`.
- **Post-run**: no cross-run comparison decision yet ("luego vemos" — we'll
  regenerate the same boxplots and add a comparison table once results are
  in). Item 5 unblocks after this.

Runbook lives in `docs/experiment-log.md`.

#### Item 4 — `bins_after=0` temporal-window ablation — Priority 2

Regenerate preprocess pickles with `bins_before=2 / bins_current=1 /
bins_after=0` (600 ms window: 400 ms past + 200 ms present, no future).

- **Target**: dual only for now.
- **Window sweep**: only `2/1/0` for now. A full ablation
  (`0/1/2`, `2/1/1`, `2/1/3`, …) is out of scope.
- **Datasets**: same 14 as the baseline (S38/S41/N59 stay excluded).
- **Results namespacing**: new top-level dir `results_bins_2_1_0/`
  (parallel to the current `results/`) so both baselines coexist for
  side-by-side comparison.

Motivation: quantify how much of the position/velocity decoding relies on
future information. A large drop vs. `after=2` would suggest the neural
code precedes behavior.

#### Item 3 — Cap `# bins` per dataset for cross-dataset comparability — Priority 3

Make the per-dataset training data quantity comparable across datasets.

- **Drop CA3-Naive-S30** (7,327 post-preprocess bins) from the batch.
- **Cap the remaining 13 datasets at 9,020 bins** (post-preprocess), the
  next minimum (Pir-Naive-N53).
- **Where the cap is computed**: post-preprocess (after every filter in
  `preprocess_data`).
- **Capping strategy — C**: keep the **last** N bins **respecting trial
  boundaries**. Walk trials backwards from the end of the recording,
  accumulating whole trials until the total lands at or just below 9,020.
  Preserves per-trial structure and the "settled behavior" regime at
  session end.
- **Effective dataset count**: 13 (14 − S30). `EXCLUDED_DATASETS_FROM_PLOTS`
  in `03-decoding-results.py` needs the extra entry.

**Open sub-decisions**:
- Where does the cap live — an option of `preprocess_data` upstream, or a
  post-preprocess script local to the thesis? (Leaning: local script for
  iteration speed; upstream later once the strategy is locked.)

#### Item 5 — Position vs. velocity dissociation analysis (phase 0 done)

Priority: after Item 1 (needs the pos-only models).

**First pass (2026-09-05, operational definition from the director):**
`exploration/11-speed-split-error.py` splits every test bin by the dataset's
median speed and compares the position error slow vs fast, stratified by
corridor zone (speed and position are correlated, so the unstratified split
confounds itself). Medians across the 17 datasets: fast decodes better inside
the corridor (MAE slow−fast: +5.1 cm pre-context, +2.8 cm in-context) while
slow decodes better past the exit (−8.0 cm — the slow post bins are reward-zone
licking, the fast ones running drift). Training on speed-filtered bins
(test 2) remains queued below.

**Phase 0 (2026-09-13, post-hoc, no GPU — the decoupling battery):**
`exploration/13-decoupling-posthoc.py` (A1 within-position speed↔error coupling;
A2 broken-stereotypy bins, |z| > 2 vs the per-position speed profile, position-matched
Δ MAE with bootstrap CI; A3 error correlation between the pos and vel decoders) and
`exploration/14-decoupling-behavioral-control.py` (ridge decoding position from the
speed window + context only — the behavioural ceiling — on the neural studies' exact
fold partition). Outputs under `results/plots/decoupling/{posthoc,behavioral-control}/`.
Headline on the six best decoders: coupling +0.00…+0.16, slow-atypical Δ MAE +0…+3 cm,
behavioural ceiling R² 0.11–0.19 vs neural 0.32–0.52 (margin +0.15…+0.38, 5/5 folds
everywhere) — the decoder carries position information well beyond the running
stereotypy, with a small residual speed leak (largest in DG_S17). Phase 1 (train-on-
one-speed-regime transfer) is optional confirmation, to decide with the director.

Post-hoc analysis, not a training run. Filter test-set bins where the
animal's velocity profile breaks its own stereotypy, then re-evaluate the
R² of the pos-only model on that subset. If R² holds, the decoder is
genuinely reading position rather than reconstructing it from velocity.

**Open questions to refine before running**:
- Operational definition of "broken stereotypy" — trial-level velocity
  mean outlier, bin-level departure from velocity(position) profile,
  specific pauses, or multi-criteria.
- Model to use — pos-single (cleaner argument, requires Item 1) vs. dual
  head (has seen velocity in training).
- Baseline of comparison — normal-subset R² vs. atypical-subset R².
- Statistics — bootstrap / CI to handle small-subset noise.
- Interpretation policy — what does a low R² in atypical bins mean:
  dissociation refuted, or the subset is out-of-distribution?
- Which datasets — only experts (stereotypy is clearest there), or all 14?

#### Item 2 — Sparse-position filter (queued)

Priority: last, most methodological refinement pending.

New preprocess filter that removes bins whose corresponding position is
under-represented in the recording. Motivation: reduce training on
positions the model can't generalize to.

**Open questions to refine before implementing**:
- Histogram bin width — 1 cm? 0.5 cm?
- Threshold — relative (< 1% of total) vs. absolute (< 5 s accumulated)
  vs. combined (AND / OR).
- Granularity — per-bin filter, or drop whole trials that touch sparse
  positions?
- Natural edge cases — should the filter respect or remove: corridor start
  (position 0, high count from every trial), reward zone (~60 cm, high
  when the animal licks), drift zone (> 60 cm, always sparse).
- Implementation location — new filter inside `ibioml.preprocess_data`
  (PR upstream, D-012 test-first) vs. thesis-local post-preprocess script.
- Impact on comparability: applying this filter invalidates all previous
  batches as a comparison; every result would have to be re-run.

---

#### Item 6 — Position-tail filter: drop bins past the context exit — Priority 1 (active)

Added 2026-08-26, agreed with the director after linear baselines run by the
lab showed the position error concentrating at both ends of the corridor.
Confirmed on the thesis MLPs by `exploration/08-position-error-analysis.py`
(single-output position studies, five test folds pooled, cm): median signed
error is **+13 cm** in the pre-context stretch, ≈ 0 inside the context, and
**−22 cm** past the context exit; 21 % of the surviving test bins lie past
the exit (62.9 cm). Hypothesis: neural activity after the context exit
reflects reward consumption / licking rather than position, so training on
those bins hurts the decoder everywhere.

**The one change vs. the shared baseline:** a new bin-level filter in
`ibioml.preprocess_data` that, per trial, removes every bin from the first
crossing of `thPositionEnd` (raw encoder units; **515° ≈ 62.9 cm — confirmed by the
director on 2026-09-05 as the exact cut used in his linear baselines**)
to the trial's last bin, padded by `bins_after` before the crossing so no
surviving sample carries post-exit activity in its history window.
Union-merged into `indices_to_remove` before standardisation. Every other
filter (teleport transitions, immobility, dPrime, out-of-corridor, neuron
filters) stays exactly as in the baseline. ADR: D-017.

- **Pilot**: `Pir_N31_expert`, `CA3_S18_expert`, `DG_S17_expert` (best
  position R² per region), **pos-only**. 3 runs ≈ 3–4 h GPU. Extend to all
  17 pos-only only if the pilot holds up under the protocol below.
- **Pickles**: `data/processed_posend515/`, via new `--th-position-end` /
  `--out-dir` flags in `scripts/preprocess_all.py`.
- **Results**: `results_posend515/`, via new `--data-dir` / `--results-dir`
  flags in `scripts/run_all_experiments.py`.
- **Evaluation protocol (D-018)**: RMSE and MAE in cm as primary metrics, R²
  secondary. Baseline and filtered models are compared **on the same bins**
  (true position ≤ 62.9 cm): the baseline restricted post hoc, the new batch
  read with `RESULTS_DIR=results_posend515 python exploration/08-…py`. The
  filtered model's R² is never compared to the baseline's full-support R².
- **Data cost**: 26–35 % of the surviving bins per pilot dataset
  (`CA3_S18` lands at 8,946 bins, below the current 9,020 minimum — relevant
  to Item 3 if the two are ever combined). Side effect: the neuron filters
  run on the surviving bins, so the variant keeps 1–2 more neurons than the
  baseline (see D-017).
- **Implementation**: fork branch off `feat/out-of-corridor-trial-filter`
  (upstream PR #12 still open, and `preprocess_all.py` already depends on
  `thPosition`), test-first (D-012), `docs/preprocessing.md` filter #9.
- **Not in scope**: the corridor start. Its bias (+13 cm) has a different
  mechanism (ambiguous neural state right after the teleport) and is already
  handled by the trial-transition filter; one change at a time.
- **Nice to have — buffer cut** (proposed 2026-08-27, parked 2026-09-05): a
  second variant with `thPositionEnd` ≈ 650° (~80 cm), evaluated on ≤ 62.9 cm
  like the pilot. Rationale: the pilot's edge effect moved to the context exit
  (bias −8 → −17 cm inside the context); a buffer keeps the far tail out of
  training while the context stays away from the training edge, separating
  "target-distribution / edge" effects from "contaminated activity" effects.
- **Post-hoc (2026-09-05)**: `exploration/10-error-by-trial-type.py` tested the
  hypothesis that rewarded+licked trials drive the tail error. They own ~62 %
  of the post-context test bins but are the *best*-decoded trials there at
  matched positions; the deep tail belongs mostly to unrewarded running-drift
  trials, whose bias plunges to −30/−45 cm. Inside the corridor the four
  ctxt × odor types have indistinguishable error curves.

#### Item 7 — Gaussian-smoothed spike counts replacing the history window — approved 2026-09-05

Director's proposal: impose temporal continuity on the input by smoothing each
neuron's spike-count series with a short Gaussian (σ = 200 ms = 1 bin), and let
that replace the 5-bin history window (`bins_before/current/after` 2/1/2 →
0/1/0, current bin only).

Pilot on **one dataset** (`Pir_N31_expert`, best position R²), pos-only, full
2×2 so each factor is attributable (agreed with Facundo 2026-09-05):

| run | activity | window | answers |
|---|---|---|---|
| baseline (exists) | raw | 2/1/2 | anchor |
| A | smoothed σ=200 ms | 0/1/0 | the proposal |
| B | raw | 0/1/0 | how much the window alone was worth |
| C | smoothed σ=200 ms | 2/1/2 | does smoothing add on top of the window |

- **Implementation**: `gaussianSigmaMs=None` (off by default) in
  `ibioml.preprocess_data`, applied per neuron along time **before**
  `get_spikes_with_history` (fork branch, test-first per D-012);
  `--gaussian-sigma-ms`, `--bins-before`, `--bins-after` flags in
  `scripts/preprocess_all.py`. Each run gets its own pickle/results tree.
- **Comparability** (measured on the smoke run, 2026-09-05): smoothing removes
  no bins, but the window size does — `bins_before/after` double as row-filter
  padding, so the 0/1/0 runs keep **20,527** bins on N31 vs the baseline's
  **18,711** (same 19 neurons everywhere). Comparison plan: **A vs B** direct
  (identical rows, verified) and **C vs baseline** direct (identical rows,
  verified); across window sizes, restrict to the common bins — the 2/1/2
  kept set is a strict subset of the 0/1/0 one, reconstructable
  deterministically from the filters.
- **Known limitation**: the Gaussian blurs activity across the boundaries of
  removed bins (teleports, immobility stretches); with σ = 1 bin the leak dies
  within ~2 bins and transitions are already padded. Recorded, accepted for
  the pilot.
- **Cost**: ~50–80 min per run on the GPU host → ~2.5–4 h for A+B+C.

### Phase 5 cross-item conventions

- **One change at a time.** Each item modifies exactly one factor vs. the
  shared baseline above. A run cannot combine two items' changes.
- **Results namespacing.** New baselines get their own top-level dir (e.g.
  `results_bins_2_1_0/`) so the SFN-poster run under `results/` stays
  intact for side-by-side reference.
- **Experiment log.** Every batch launched adds a row to
  `docs/experiment-log.md` with date, config, datasets, results dir.
- **ADRs on methodological items.** Items 2, 3 and 4 warrant an ADR each
  once their sub-decisions are locked.

## Backlog (things intentionally deferred)

- Config-driven experiment runner
- Cross-experiment result aggregation / comparison dashboard
- Upstreaming fixes to ibioml PyPI release
- CI (linting, tests on PR)
- MkDocs site (if the lab adopts the project)
