# `ibioml` Audit

**Audit target:** `external/ibioml/` submodule, branch `thesis/audit-baseline`,
pointing at upstream commit `4e477c3` (Release v0.1.6).

**Audit date:** 2026-04-17. **Auditor:** Facundo + Claude.

**Scope.** Read every source module, run the bundled "test suite," and document
bugs, code-quality issues, and extension opportunities. Prioritize findings for
upstream fixes.

---

## TL;DR

- **One confirmed live bug** in `preprocess_data` (reproduced in a throwaway run) —
  this is the index-type bug the marinburginlab scripts already monkey-patch.
- **Effective test coverage: zero.** The four files `pytest` discovers either
  print-without-asserting or return bools that pytest warns about.
- **Version/metadata inconsistency:** `__init__.py` says `0.1.2`, `setup.py` says
  `0.1.6`, release tag is `v0.1.6`. `setup.py` has a `url="https://github.com/tuusuario/IBioML"`
  placeholder.
- **Global mutable state in `models.py`:** a module-level `device` is used inside
  every model's `forward()`, ignoring the `device` the trainer passed at
  construction time. Real correctness risk on mixed CPU/GPU setups.
- **R² is computed two different ways** (training-time vs final evaluation).
  Training-time formula is subtly wrong because MSE and `torch.var` use different
  denominators. Final R² via sklearn is correct.
- **`inner_folds` parameter is accepted but never used.** The "nested CV" is
  actually one outer CV loop over trials + a single inner train/val split.
  Misleading API.
- **`preprocess_data` hardcodes `data/` as the output directory.** Cannot be run
  from a different working directory without silently writing to the wrong place.
- **Spanish docstrings and `print`-based logging everywhere** — usability issue
  for an English-language thesis.

Fix priorities in [§6 Prioritized fix list](#6-prioritized-fix-list).

---

## 1. Module map

```
external/ibioml/
├── ibioml/
│   ├── __init__.py               33 lines  — public API
│   ├── models.py                 345 lines — nn.Module classes (MLP/RNN/GRU/LSTM + dual-output)
│   ├── trainer.py                546 lines — BaseTrainer + Single/DualOutputTrainer + train_model()
│   ├── tuner.py                  399 lines — Tuner class + run_study() wrapper (Optuna + nested CV)
│   ├── preprocess_data.py        234 lines — load .mat → 12 pickles (with/without context × 3 targets × seq/flat)
│   ├── plots.py                  254 lines — top-level plotting helpers (load_results, boxplots)
│   └── utils/
│       ├── __init__.py           21 lines
│       ├── data_scaler.py        293 lines — StandardScaler wrappers (2D/3D feature, single/dual target)
│       ├── evaluators.py         187 lines — SingleOutputEvaluator, DualOutputEvaluator (final-metric calculation)
│       ├── splitters.py          156 lines — TrialKFold, trial_train_test_split
│       ├── trainer_funcs.py      148 lines — initialize_weights, create_dataloaders, EarlyStopping, configure_train_config
│       ├── tuner_funcs.py        102 lines — split_model_space, initialize_config, make_serializable
│       ├── preprocessing_funcs.py 145 lines — bin_spikes, bin_output, get_spikes_with_history, create_trial_markers
│       ├── model_factory.py      33 lines  — create_model_class (single vs dual)
│       ├── pipeline_utils.py     42 lines  — find_stationary_window, map_to_discrete_hidden_size (orphaned)
│       ├── plot_functions.py     389 lines — enhanced styled plots (parallel to plots.py)
│       └── plot_styles.py        51 lines  — matplotlib/seaborn defaults
├── test_imports.py               117 lines — runs imports, returns bool (not asserts)
├── test_exclusive_params.py      251 lines — demo script showing dual-output model structures
├── setup.py                      62 lines
├── MANIFEST.in, LICENSE, README.md, mkdocs.yml, release.sh, docs.sh
├── docs/                         (mkdocs site scaffolding)
└── examples/                     (notebooks, sample outputs)
```

Total source: ~2,500 lines across 15 modules. Monolithic files
(`trainer.py` at 546 lines exceeds the 500-line quality threshold).

### Public API (from `ibioml/__init__.py`)

| Import | What it is |
|---|---|
| `ibioml.MLPModel` / `RNNModel` / `LSTMModel` / `GRUModel` | Single-output nn.Module |
| `ibioml.models` | Full module (also has `DualOutput*` classes not in `__all__`) |
| `ibioml.preprocess_data` | Module; main entry: `preprocess_data(file_path, ...)` |
| `ibioml.tuner` | Module; main entry: `run_study(X, y, T, model_space, ...)` |
| `ibioml.trainer` | Module; main entry: `train_model(config)` |
| `ibioml.plots` | Module; `load_results`, `extract_r2_scores`, `boxplot_test_r2_scores` |

**Undocumented extensions:** `DualOutputMLPModel`, `DualOutputRNNModel`,
`DualOutputGRUModel`, `DualOutputLSTMModel` exist but aren't in `__all__`. They're
used internally by `model_factory.create_model_class` when `y.shape[1] == 2`.

---

## 2. Test coverage — effectively zero

### What's there

| File | What it does | Real assertion count |
|---|---|---|
| `test_imports.py` | Imports every module; returns `True`/`False` | 0 |
| `test_exclusive_params.py` | Instantiates dual-output models and prints their structure | 0 |

Running `pytest external/ibioml/ -q`:

```
4 passed, 2 warnings in 7.81s
```

Both warnings are `PytestReturnNotNoneWarning`: test functions return booleans
instead of asserting.

### Verdict

- **No behavioral tests exist.** If someone changes the R² formula, silently
  breaks `trial_train_test_split`, or rearranges `preprocess_data` output paths,
  nothing in the current suite will notice.
- The two demo scripts are useful as examples but should move to `examples/`,
  not be named `test_*.py`.
- Creating a real `tests/` directory with unit + integration tests is a
  prerequisite for any non-trivial fix.

### Bundled example

`examples/results/mlp_nested_cv/study_2025-05-20_13-07-37/` contains a pre-run
artifact tree (5 folds × 2 trials). This is useful *regression fixture* material —
we could compare future refactor outputs against it. Don't delete.

---

## 3. Live-run verification

To confirm findings aren't purely static, I ran the default naive-cohort config
end-to-end against `data/raw/L7_naive_bins200ms.mat`:

```python
preprocess_data(
    file_path='.../L7_naive_bins200ms.mat',
    file_name_to_save='L7_naive_bins200ms',
    bins_before=5, bins_after=5, bins_current=1,
    threshDPrime=-1.1, firingMinimo=100,
)
```

**Result:**

```
...
Trials con dPrime menor o igual a 2.8: []   ← stale hardcoded message (bug #3)
Índices totales de los time bins a eliminar:
  [0.0000e+00 1.0000e+00 2.0000e+00 ... 1.7489e+04 1.7490e+04 1.7491e+04]
                                                                           ↑
                                              FLOAT array, not int
IndexError: arrays used as indices must be of integer (or boolean) type
```

- Float-indexed array confirms bug #1 (index-type).
- Stale "2.8" string confirms bug #3.
- Bug also implicitly confirms bug #2 (hardcoded output path): nothing was
  written, because the error occurred before `save_data`.

---

## 4. Findings — bugs

Numbering is stable; used in the prioritized fix list (§6) and in future commit
messages. "Severity" uses: **blocker** (breaks core use), **major** (incorrect
results or usability hole), **minor** (cleanup).

### #B-01 — `preprocess_data` index-type bug **[BLOCKER, reproduced]**

**File:** `ibioml/preprocess_data.py`, lines 149–150.

```python
rmv_time = np.where(np.isnan(y[:,0]))            # returns (array,) — a tuple
indices_to_remove = np.union1d(rmv_time,         # ↓ tuple + array → dtype coerces to float
                               np.union1d(indices_to_remove_temp,
                                          indices_to_remove_low_performance))
```

`np.where(cond)` returns a tuple `(array,)`. Passing that tuple through
`np.union1d` causes NumPy to coerce the result to `float64`, and then
`np.delete(X, indices_to_remove, 0)` fails with `IndexError: arrays used as
indices must be of integer (or boolean) type`.

**Already has a known workaround:** `marinburginlab-labPrograms/.../san/all-datasets-experiment.py`
contains a local `preprocess_data_fixed` that casts `.astype(int)` on every
index array.

**Fix:**
```python
rmv_time = np.where(np.isnan(y[:, 0]))[0]  # unwrap tuple
indices_to_remove = np.union1d(
    rmv_time,
    np.union1d(indices_to_remove_temp, indices_to_remove_low_performance),
).astype(int)
```

**Test:** run preprocessing on a naive dataset with the parameters that
currently fail.

### #B-02 — Hardcoded output directory in `preprocess_data` **[MAJOR]**

**File:** `ibioml/preprocess_data.py`, lines 169–189.

Every `save_data` call embeds `'data/' + file_name_to_save + '...'`. The `data/`
prefix is a hardcoded relative path. Run from any CWD other than "a directory
that contains a `data/` subdirectory," files silently go to the wrong place
or `FileNotFoundError`.

**Fix:** accept `output_dir: str | Path = "data"` as a parameter. Create it with
`os.makedirs(output_dir, exist_ok=True)` before writing. Use `os.path.join`.

### #B-03 — Stale hardcoded threshold in log message **[MINOR]**

**File:** `ibioml/preprocess_data.py`, line 74.

```python
print("Trials con dPrime menor o igual a 2.8:", low_performance_trials_indices)
```

Message hardcodes `2.8` but the actual threshold is the parameter `threshold`.
Misleading console output.

**Fix:** `print(f"Trials with dPrime ≤ {threshold}:", ...)`.

### #B-04 — `rewOdor` silently dropped in `load_data` **[MAJOR]**

**File:** `ibioml/preprocess_data.py`, line 30.

```python
return mat_contents, neural_data, rewCtxt, trialFinalBin, dPrime, criterion, rewCtxt, pos_binned, vels_binned
#                                   ^^^^^^^                                   ^^^^^^^
#                                   Duplicate! The second one should be rewOdor.
```

The caller unpacks nine values, with `rewCtxt` assigned twice. The variable
names and positions strongly suggest the second `rewCtxt` was meant to be
`rewOdor` (loaded but never read into a local → silently dropped).

Downstream: `preprocess_data` never uses `rewOdor` information, so this hasn't
caused visible harm — but it means rewarded-vs-unrewarded-odor analyses are
impossible without a separate load. Scientific gap, not just code gap.

**Fix:** load `rewOdor` and return it. Remove the duplicate `rewCtxt`. Update
all call-sites (check there's only one: `preprocess_data` itself).

### #B-05 — Off-by-one asymmetry in trial-bin removal helpers **[MAJOR]**

**Files:** `ibioml/preprocess_data.py`.

```python
# Line 63 — get_idx_by_high_trial_duration uses EXCLUSIVE range
indices_to_remove_trialDuration.extend(range(startInd, endInd))

# Line 86 — get_idx_by_low_performance uses INCLUSIVE range
indices_to_remove_low_performance.extend(range(startInd, endInd+1))
```

Both helpers claim to return "all time bins belonging to the flagged trial."
One of them is wrong — they should agree.

`create_trial_markers` (in `utils/preprocessing_funcs.py`, line 143) uses
`trial_markers[start_idx:end_idx+1] = trial_idx` — i.e., inclusive. That's the
semantic truth: `trialFinalBin[i]` is inclusive.

**Conclusion:** `get_idx_by_high_trial_duration` is the bug. It misses the last
bin of each too-long trial.

**Fix:** change line 63 to `range(startInd, endInd+1)`.

### #B-06 — Module-level `device` in `models.py` overrides caller's choice **[MAJOR]**

**File:** `ibioml/models.py`, line 5.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Every `forward()` does `x = x.to(device)` using this module-level global.
The trainer (`trainer.py:67`) calls `.to(config["device"])` on the model
*weights* using the caller-supplied device — but `forward` then .to()'s the
input using the module-level device.

**Consequence:** if a user on a CUDA-capable machine passes
`device="cpu"` in `model_space`, inputs still move to CUDA and weights stay
on CPU → `RuntimeError: Expected all tensors to be on the same device`.

Also: the import-time `set_global_seed()` on line 24 is a silent side effect —
simply importing `ibioml` reseeds PyTorch's global RNG.

**Fix:** store `self.device` on the model at construction; use it in `forward`.
Remove the module-level global and the import-time `set_global_seed()` call.
Expose `set_global_seed` as an explicit call users opt in to.

### #B-07 — `inner_folds` parameter advertised but unused **[MAJOR]**

**File:** `ibioml/tuner.py`.

`run_study` accepts `inner_folds=5`. `Tuner.set_study_params` stores it.
`_run_outer_cv_loop` never uses it — instead it performs a single
`trial_train_test_split(train_size=0.8, ...)` per outer fold (line 114).

User scripts that pass `inner_folds=1` happen to get what they expect (one
inner holdout). Anyone passing `inner_folds=5` silently gets the same one
split, not a 5-fold inner CV.

**Fix:** either (a) actually implement inner k-fold CV (and pick the best
config by mean-over-inner-folds), or (b) rename the parameter to
`inner_split_ratio` / drop it. Prefer (a) for a true nested-CV guarantee.

### #B-08 — Training-time R² formula is slightly wrong **[MINOR]**

**File:** `ibioml/trainer.py`, lines 331 and 464–465.

```python
r2_score = 1 - val_loss.item() / torch.var(y_val_tensor).item()
```

`val_loss` uses `nn.MSELoss()` with default `reduction='mean'` (denominator `N`).
`torch.var` defaults to **unbiased** (denominator `N-1`). So:

```
code:   1 − (Σ(y-ŷ)²/N) / (Σ(y-ȳ)²/(N-1))
truth:  1 − Σ(y-ŷ)²/Σ(y-ȳ)²
```

For large `N` the difference is negligible, but the per-epoch R² that's logged
and used as the Optuna objective is slightly biased. The **final-reported** R²
in `evaluators.py` uses `sklearn.metrics.r2_score` and is correct — so two
different numbers coexist.

**Fix:** use `torch.var(y_val, unbiased=False)`, or switch entirely to sklearn's
`r2_score` inside the trainer for consistency.

### #B-09 — `configure_train_config` references nonexistent attribute **[MAJOR, latent]**

**File:** `ibioml/utils/trainer_funcs.py`, line 109.

```python
train_config["y_scaler"] = scaler.y_scaler
```

`scaler` is a `DataScaler` object, which has `feature_scaler` and `target_scaler`
— no `y_scaler`. Any call path that hits this function will crash.

Where it's called from: `BaseTrainer.train_with_cv`, line 304. That method is
only used when `config["n_splits"]` is set. The default `run_study` path does
not set `n_splits`, so this is latent. If anyone ever passes `n_splits` in the
model_space, it blows up.

**Fix:** `train_config["y_scaler"] = scaler.target_scaler` (or drop the
attribute entirely; the scaler itself is already stored as `train_config["scaler"]`
by `initialize_config`).

### #B-10 — `create_dataloaders` drops the last batch unconditionally **[MINOR]**

**File:** `ibioml/utils/trainer_funcs.py`, line 77.

`drop_last=True` on the training DataLoader. For datasets with ~2,000 samples
and batch_size 32, that's ~1.5% data loss every epoch. For smaller trial-split
subsets it matters more.

**Fix:** expose as parameter, default `False` for val, `True` only if the user
opts in.

### #B-11 — `torch.load(..., weights_only=False)` is unsafe-by-default in modern PyTorch **[MINOR, future]**

**File:** `ibioml/tuner.py`, line 161.

PyTorch 2.4+ deprecates `weights_only=False` and future versions plan to make
`weights_only=True` the default. Since our models are pickled Python objects
(saved via `torch.save(model, ...)` not state_dict), loading them
`weights_only=True` won't work.

**Fix now:** OK. **Fix later:** save `model.state_dict()` + a separate config
describing how to reconstruct the architecture. Then `torch.load` can be safe.

### #B-12 — `map_to_discrete_hidden_size` hardcodes `[64, 128, 256, 512]` **[MINOR, orphaned]**

**File:** `ibioml/utils/pipeline_utils.py`, line 42.

Hardcoded discrete set. This function doesn't appear to be called anywhere in
the package (grep: unused). Dead code.

**Fix:** delete or parametrize.

### #B-13 — `setup.py` URL is a placeholder **[MINOR]**

**File:** `setup.py`, line 46.

```python
url="https://github.com/tuusuario/IBioML",
```

"tuusuario" is Spanish for "your-user" — clearly a template placeholder that
was never filled in. PyPI therefore advertises a broken URL.

**Fix:** `url="https://github.com/mariburginlab-labPrograms/IBioML"`.

### #B-14 — Version string mismatch between `__init__.py` and `setup.py` **[MINOR]**

- `ibioml/__init__.py`: `__version__ = "0.1.2"`
- `setup.py`: `version="0.1.6"`
- Latest tag: `v0.1.6`

**Fix:** single source of truth (either read `__version__` from setup, or use
`importlib.metadata` to derive at runtime).

### #B-15 — Wildcard imports hide provenance **[MINOR]**

**Files:** `ibioml/tuner.py:12`, `ibioml/trainer.py:14`.

```python
from ibioml.utils.tuner_funcs import *
from ibioml.utils.trainer_funcs import *
```

Makes it impossible to tell without searching where `make_serializable`,
`initialize_config`, `split_model_space`, `EarlyStopping`, `create_dataloaders`
come from. Also pollutes namespace.

**Fix:** explicit imports.

### #B-16 — `report_metrics` truncates to shortest metric list, silently losing data **[MINOR]**

**File:** `ibioml/trainer.py`, `_save_final_results`, line 222.

```python
min_length = min(len(v) for v in self.metrics.values() if isinstance(v, list))
adjusted_metrics = {k: v[:min_length] if isinstance(v, list) else v ...}
```

For `DualOutputTrainer`, some metrics are appended in `train_one_epoch`
(e.g., `train_loss_pos`), others in `evaluate_model` (e.g., `val_loss_pos`).
If training halts mid-epoch (early stop after a val phase, for example), the
lengths differ and this silently drops the last entries of the longer lists.

**Fix:** pad with NaNs to max length, or guarantee paired appends via the
trainer loop.

### #B-17 — `torch.load(.pt)` fragile to path collision in repeated runs **[MINOR]**

**File:** `ibioml/tuner.py`, lines 84, 160.

When a custom `study_name` is reused, `os.makedirs(..., exist_ok=True)` silently
overwrites. `best_model.pt` from a prior run gets overwritten during the next
run's objective function. If the new run fails before producing a best model,
`torch.load(best_model_path)` may load a stale model from a previous run.

**Fix:** refuse to reuse an existing non-empty study directory, or timestamp
every `study_name` regardless.

---

## 5. Code-quality concerns (not bugs, but friction)

- **Spanish docstrings and comments everywhere.** Thesis is English; upstream
  audience is international. All of `ibioml/` should be translated.
- **`print`-based logging.** No way to silence or redirect; hardcoded `🔧/✅/❌`
  emoji prefixes in user-facing scripts. Should use the `logging` module.
- **Two parallel plotting modules** (`plots.py` top-level and
  `utils/plot_functions.py`) with overlapping responsibilities. Pick one, delete
  or clearly demarcate the other. `plots.py` also uses `from IPython.display
  import display, HTML` at module top — forces IPython even in non-notebook use.
- **`Tuner` API is half-fluent / half-kwargs.** `set_model_space()`,
  `set_study_params()`, `set_evaluator()` return `self`, suggesting chaining,
  but the `run()` method takes `(X, y, T)` directly and `__init__` already
  takes model_space + evaluator. Inconsistent.
- **Global `palette` / `color_dict` / `medians` in `plots.py`** (lines 11–14).
  Module-level mutable state used across functions.
- **`BaseTrainer.__init__` conditional model init** (`trainer.py:24`): skips
  `initialize_model` unless `train_loader` and `val_loader` are already in
  config. Easy to forget → silent "uninitialized trainer" state.
- **Six+six output files** from `preprocess_data` (with/without context ×
  position/velocity/both × seq/flat). Storage blow-up, and every downstream
  consumer has to know which filename variant to load. A single output with
  a selector argument would be cleaner.
- **File length.** `trainer.py` at 546 lines exceeds the 500-line quality
  cap; `tuner.py` at 399 is borderline.
- **`match` statement** in `utils/trainer_funcs.py:17` requires Python 3.10+,
  but `setup.py` declares `python_requires=">=3.8"`. Either raise the minimum
  or refactor to `if/elif`.

---

## 6. Prioritized fix list

Ordered by: blocker → major → minor, breaking changes before compatible ones,
and dependency chains (fix underlying bugs before the changes that rely on them).

### Tier 1 — fix immediately (blocker / reproduces in our data)

1. **#B-01** `preprocess_data` index-type bug. Also remove the marinburginlab
   monkey-patch once the upstream fix lands. **(Straight into upstream.)**

### Tier 2 — fix this audit cycle (correctness)

2. **#B-06** Module-level `device` in `models.py`. Store on the model.
3. **#B-05** Off-by-one in `get_idx_by_high_trial_duration`.
4. **#B-04** Drop-or-restore `rewOdor` in `load_data`.
5. **#B-09** `configure_train_config` → `scaler.target_scaler`.
6. **#B-07** Either implement true inner k-fold CV, or rename/document
   `inner_folds` as a single-split ratio.

### Tier 3 — usability, consistency

7. **#B-02** Parameterize output directory in `preprocess_data`.
8. **#B-08** Fix training-time R² denominator (or switch to sklearn).
9. **#B-03** Fix stale "2.8" log message.
10. **#B-13** / **#B-14** setup.py URL + version string.
11. **#B-15** Explicit imports in tuner/trainer.
12. **#B-16** Consistent metric-list lengths.
13. **#B-17** Prevent silent overwrites across repeated study runs.
14. **#B-10** Expose `drop_last` to the user.

### Tier 4 — cleanup / future

15. **#B-12** Remove or parameterize `map_to_discrete_hidden_size`.
16. **#B-11** Plan migration to `state_dict`-based checkpoints.
17. Translate Spanish → English across all of `ibioml/`.
18. Replace `print` with `logging`.
19. Split `trainer.py` to stay under the 500-line cap.
20. Consolidate `plots.py` + `utils/plot_functions.py`.

Each fix gets a dedicated branch on the fork (`fix/B-01-preprocess-index-type`,
etc.) and a test in `external/ibioml/tests/` (create the directory — none exists
today). See [`specs/fork-ibioml.md`](../specs/fork-ibioml.md) for the PR flow.

---

## 7. Feature extension ideas (deferred — out of Phase 1 scope)

Not bugs, but candidate features that came up while reading:

- **Stratified trial split by duration / performance** — the legacy project's
  `TRIAL_BASED_SPLITTING.md` describes this; `splitters.py` does unstratified
  shuffles.
- **Configurable loss weighting** in dual-output models. Currently
  `loss = loss_pos + loss_vel` equally. Users might want to weight position
  over velocity when velocity has much lower variance.
- **Checkpointing & resumption.** Long neuron-scaling runs (24–38h) have no
  ability to resume from a checkpoint if interrupted.
- **Explicit seed API.** `GLOBAL_SEED = 42` buried in `models.py`; `run_study`
  uses `random_state=42` in several places without exposing it as a parameter.
- **Optuna study persistence.** `optuna.create_study()` creates an in-memory
  study every time — no `storage=sqlite:///...` option exposed. Can't inspect
  search trajectories offline.
- **Cross-dataset transfer API.** The marinburginlab
  `transfer-learning-neuron-scaling.py` implements this outside the library; it
  could be a first-class `ibioml.tuner.run_transfer_study(...)`.
- **Batch-aware `torch.var` for R².** When switching to `unbiased=False`,
  consider also supporting multi-output R² with sklearn's `multioutput` kwarg.
- **Linear / ridge / SVR baselines.** Currently only neural models. A
  `LinearBaseline` in `models.py` would provide a calibration floor for R².

---

## 8. Upstream-contribution plan

- **Priority PRs (Tier 1 + 2) go back to `mariburginlab-labPrograms/IBioML`**
  one branch per bug, each with a regression test.
- **Tier 3/4** can accumulate on the fork first; batch-PR them once the audit
  dust settles.
- **Breaking API changes** (e.g., renaming `inner_folds`) need coordination
  with Facundo's collaborators who use the library in the lab. Flag them before
  opening PRs.
- **Creation of `tests/`** is a standalone PR, merged before any behavioral PR
  so that subsequent fixes can include tests.

---

## 9. What this audit did NOT do

- Static-analysis passes (ruff/mypy) against the codebase — worth running once
  the fork is set up for editing.
- Security scans — unlikely to surface much on a scientific library but
  `gitleaks` would catch any accidentally-committed secrets.
- Benchmarking — no attempt to measure runtime or memory.
- `docs/` audit — only skimmed.

Follow-ups for a future session if they become relevant.
