# Research — State of the Neural Decoding Project (pre-restructure)

Everything below was gathered from the three source directories before any
changes were made. This is a frozen snapshot to give future sessions instant
context. Dates refer to last-modified times observed on disk.

## 1. Three directories at a glance

| Directory | Purpose | Last active | Quality |
|---|---|---|---|
| `neural-decoding/` | Exploratory, lab-notebook style | April 2025 | Messy, one-off |
| `neural-decoding-ml/` | Clean-rewrite attempt (OOP) | Aug/Oct 2025 | Modular but stalled |
| `marinburginlab-labPrograms/neural-decoding/` | Production experiments on top of `ibioml` | Oct/Nov 2025 | Flat scripts, real results |

All three target the same scientific problem: **decoding mouse position (and velocity)
from neural activity** recorded during an associative odor/context task, across
multiple brain regions.

## 2. Scientific problem

Mice navigate a virtual-reality corridor with odor cues. Neural activity is recorded
in either piriform cortex (L3, L5, L7, N31, N53, N59 — expert and naive animals)
or hippocampus (CA3_S18, CA3_S30, CA3_S31, DG_S19, DG_S24, DG_S38, DG_S39, DG_S41).
The decoding target is the animal's position within the corridor; secondary target
is velocity. Some experiments decode both jointly (`output_size=2`).

## 3. Common data model

Every `.mat` file exposes the same variables (shape conventions in comments):

- `neuronActivity` — 3D, `(T_bins_total, N_neurons)` in recent files; some older
  files are `(E, N, T)` with E = trials. **This shape inconsistency is important.**
- `position` — 2D `(T, 1)` (older docs claim `(E, T)`)
- `speed` or `velocity` — 2D `(T, 1)` — **name inconsistency across datasets** (`speed`
  in the docs/data-structure.md of neural-decoding-ml, `velocity` in the
  marinburginlab `all-datasets-experiment.py`)
- `rewCtxt` — 1D `(E,)`, 1 if context is rewarded
- `rewOdor` — 1D `(E,)`, 1 if odor is rewarded
- `goResponse` — 1D `(E,)`, 1 if mouse licked
- `dPrime` — 1D `(E,)`, cumulative performance up to that trial
- `criterion` — 1D `(E,)`, lick tendency bias
- `trialDurationInBins` — 1D `(E,)`
- `trialFinalBin` — 1D `(E,)`, last temporal bin of each trial

### Preprocessing pipeline (as implemented in `ibioml.preprocess_data`)

1. Load `.mat`, reshape.
2. Add context features (rewCtxt + rewCtxt_neg) as extra "neurons" → 2 context
   columns appended to the neuron dimension.
3. `get_spikes_with_history(bins_before, bins_current, bins_after)` — standard
   history window. Typical config: 5+1+5 = 11 history bins.
4. Clean boundary bins (first `bins_before`, last `bins_after`).
5. Remove trials with excessive duration (`get_idx_by_high_trial_duration`).
6. Remove trials with `dPrime < threshDPrime` (2.5 for expert, -1.1 or -3.0 for naive).
7. Drop NaN positions.
8. Drop low-firing neurons (`firingMinimo`: 1000 for expert, 100 for naive, 50 extreme).
9. Save six artifacts per config:
   - `{name}_withCtxt.pickle` (sequential)
   - `{name}_withCtxt_flat.pickle` (flattened for MLP)
   - `{name}_withCtxt_onlyPosition.pickle` / `_flat`
   - `{name}_withCtxt_onlyVelocity.pickle` / `_flat`
   - `{name}.pickle` / `_flat` (no context)

### Pickle format evolved over time

- **Old:** `(X, _, y)` — 3 elements, middle slot unused
- **Current (neural-decoding-ml):** `(X, y, trial_ids)` — 3 elements
- **ibioml (marinburginlab):** `(X, y, T)` — 3 elements, `T` = trial markers

Names differ; semantics are the same. `neural-decoding-ml/baseline_trainer.py` has
branching logic to handle both formats (`_load_data`), but `docs/current-status.md`
explicitly says `trial_ids` sometimes ends up assigned to `self.y_*`. **Known bug.**

## 4. Dataset catalog (rough; full list in `dataset-inventory.md`)

Datasets found across the three directories (deduplication pending):

**Expert piriform (200ms bins):** L5, L3, N31
**Expert piriform (100ms bins):** L5, L3, N31, N59
**Naive piriform (200ms bins):** L7, N53, N59
**Hippocampus (200ms bins):** CA3_S18, CA3_S30, CA3_S31, DG_S19, DG_S24, DG_S38,
DG_S39, DG_S41

The most complete dataset collection is in
`marinburginlab-labPrograms/neural-decoding/notebooks/experiments/sfn/datasets/`.

## 5. Directory deep-dives

### 5.1 `neural-decoding/` (first version)

- Root has ~15 one-off Jupyter notebooks covering all phases of development
  (exploration, hyperparameter tuning, visualization, multiple decoders).
- `models.py` (~30KB) — all models in one file.
- `utils/` — `data_scaler.py`, `pipeline_utils.py`, `preprocessing_funcs.py`, `trainer_funcs.py`.
- `data/` — preprocessed pickles for L5 and CA3_S18 (withCtxt format).
- `results/` — results for many decoder families: dnn, gru, lstm, rnn, svr, wc, wf, xgb.
- Interesting standalone scripts: `khipu-pipeline.py`, `time_series_CV.ipynb`,
  `plot_figures_poster.ipynb` (~3.4MB — for a poster).
- **Role:** idea exploration. Not a maintainable codebase.

### 5.2 `neural-decoding-ml/` (stalled rewrite)

- `models/` — good OOP: `BaseModel`, `MLP`, `RNN`, `LSTM`, `model_factory.py`, `configs.py`.
- `training/` — two overlapping trainers: `baseline_trainer.py` and
  `baseline_trainer_temporal.py` (the second is the correct one, no temporal leakage).
- `utils/` — `data_utils.py` (has `create_trial_based_data_loaders`),
  `temporal_data_utils.py`, `preprocessing_funcs.py`.
- `preprocess_data_in_bins.py` — monolithic script at root.
- `docs/current-status.md` documents known issues (frozen here):
  1. `_load_data()` mis-assigns `trial_ids` → `y`.
  2. `trial_ids_flat`/`trial_ids_seq` not exposed for trial-based splitting.
  3. `run_baseline_trial_based.py` expects API that trainer doesn't provide.
  4. Duplicate trainers with shared logic not factored.
  5. `training/train_model.py` assumes `.npy` files that don't exist — dead code.
  6. `preprocess_data_in_bins.py` is monolithic.
- `docs/oop-strategy.md` outlines a planned 5-phase refactor
  (NeuralData → PreProcessor → Splitter/DataModule → unified Trainer → declarative
  configs → orchestrator). Only Phase 0 partly implemented.
- `docs/TRIAL_BASED_SPLITTING.md` — detailed writeup of why trial-based splitting
  is the correct approach (avoids temporal data leakage inflating R² by ~0.06).
- **Role:** learning exercise to internalize ML concepts. Good architecture, bad completion.

### 5.3 `marinburginlab-labPrograms/neural-decoding/` (production)

- Flat `notebooks/experiments/` with subfolders per congress/experiment:
  - `san/` — Sociedad Argentina de Neurociencias. Contains `all-datasets-experiment.py`
    (monolithic 1000-line script running preprocessing + 9-dataset MLP nested-CV sweeps),
    plus plot scripts (`all-datasets-plot.py`, `all-datasets-plot-predictions.py`),
    and SVG outputs.
  - `sfn/` — Society for Neuroscience. Has an expanded `datasets/` (largest collection).
  - `neuron-scaling/` — well-documented 51-experiment design (1→48 neurons, 5 repetitions
    per step, nested CV with 5 outer / 1 inner fold, 10 Optuna trials) on L5 expert 200ms.
    Full protocol in `NEURON-SCALING.md`.
  - `transfer-learning-neuron-scaling.py` + `transfer-learning-synthetic.py` — newer
    (not yet run or not yet analyzed).
- Everything depends on `ibioml` (imported as `from ibioml.models import MLPModel`,
  `from ibioml.tuner import run_study`, `from ibioml.preprocess_data import preprocess_data`).
- **Critical finding:** `all-datasets-experiment.py` contains a locally-defined
  `preprocess_data_fixed` function that re-implements `ibioml.preprocess_data` to
  work around a bug: *"arrays used as indices must be of integer type"*. The fix
  is explicit `.astype(int)` casts on `rmv_time`, `indices_to_remove_temp`,
  `indices_to_remove_low_performance`, and the final `np.union1d` result.
  **This bug must be fixed in the forked ibioml during Phase 3.**
- **Role:** the real ongoing experimental work. Production quality on the outside
  (congress posters, published results), but no modular structure underneath.

## 6. `ibioml` library (local audit pending)

Located at `/Users/facundo/Development/Academia/IBioBA/Libreria/IBioML/`.

- Upstream: `https://github.com/mariburginlab-labPrograms/IBioML.git` (note the typo
  — "mari" not "marin" — in the GitHub org name).
- Latest commit: `1362805 Merge pull request #6 ... feature/split-trials-seed`.
- Structure (from `ls`):
  - `ibioml/` (package source, 11 items)
  - `docs/` (mkdocs site)
  - `examples/`
  - `tests/` — unknown state
  - `dist/`, `ibioml.egg-info/`, `setup.py`, `mkdocs.yml`, `requirements-docs.txt`
- Published to PyPI (https://libraries.io/pypi/ibioml).
- Known user-facing API:
  - `ibioml.models.MLPModel` (and likely other models).
  - `ibioml.tuner.run_study(X, y, T, model_space, num_trials, outer_folds, inner_folds, save_path)`.
  - `ibioml.preprocess_data.preprocess_data(file_path, file_name_to_save, bins_before, bins_after, bins_current, threshDPrime, firingMinimo)`.
  - `ibioml.preprocess_data.{load_data, add_context_to_data, get_idx_by_high_trial_duration, get_idx_by_low_performance, clean_neurons_by_low_firing_rate, save_data}`.
  - `ibioml.utils.preprocessing_funcs.{get_spikes_with_history, create_trial_markers}`.

Facundo is listed as author/editor with write access. Fork strategy: create
personal fork `facundomontiel/IBioML`, track upstream, contribute fixes back.

## 7. Cross-cutting technical themes

1. **Trial-based splitting is the correct approach.** Time-bin splitting leaks
   temporal correlations across train/val/test. Confirmed by both
   `neural-decoding-ml/docs/TRIAL_BASED_SPLITTING.md` and the `ibioml` tuner
   (which uses trial markers `T`).
2. **Nested cross-validation** is standard in production experiments (outer loop
   for evaluation, inner loop for hyperparameter search via Optuna).
3. **History windows** of 11 bins (5 before + 1 current + 5 after) are the norm.
   `L5_bins200ms_short_history` uses 5 bins (2+1+2) — one documented alternative.
4. **Context features** (rewCtxt, rewOdor) are appended as "virtual neurons" before
   history-windowing, so each also gets 11 history bins → +22 context features.
5. **Dual-target decoding** (position + velocity) via `output_size=2` is standard
   in the latest marinburginlab experiments.
6. **Reproducibility discipline** — `CV_FOLD_SEED=42` shared across experiments,
   per-experiment neuron-selection seeds, documented in `NEURON-SCALING.md`.

## 8. Outstanding questions (deferred)

- What does the `ibioml` tuner actually implement? (audit will tell)
- What early-stopping logic is used? (audit will tell)
- Is regularization supported in `ibioml`? The `mlp_base_space` dict has
  `reg_type=None, lambda_reg=None` slots — audit to confirm support.
- Are RNN/LSTM models in `ibioml` or only MLP? The neural-decoding-ml project had
  them, but recent marinburginlab scripts only import `MLPModel`.
- Are there pickled model checkpoints we should preserve, or are all results
  derivable from code + data + seeds?
