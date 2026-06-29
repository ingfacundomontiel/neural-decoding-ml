# Preprocessing pipeline

This document describes the bin-level and neuron-level filtering applied to
the raw `.mat` recordings before they feed the decoder. It is the analogue
of the "experiment-specific preprocessing" layer noted in
[`data/raw/README.md`](../data/raw/README.md): what *this project* adds on
top of the rig software's upstream processing, mirroring the pipeline that
Sebas runs in his own scripts.

Related documents:

- [`trial-structure.md`](./trial-structure.md) — corridor geometry and
  trial flow, the source of the assumptions used here.
- [`ibioml-audit.md`](./ibioml-audit.md) — what the lab's `preprocess_data`
  function did at audit time (the version the thesis was audited against).

## Pipeline overview

Seven distinct filters are applied to each recording. Five remove **time
bins** (rows of the feature matrix `X`); two remove **neurons** (columns
of `X`). All seven contribute to a single `indices_to_remove` list (for
rows) or `neurons_to_remove` list (for columns), which are applied **after**
`get_spikes_with_history`.

| # | Filter | What it removes | Axis |
|---|---|---|---|
| 1 | Immobility periods | bins of sustained low velocity (`velocity < thVel` for longer than `thDur`; defaults `thVel = 1 cm/s`, `thDur = 4 s`) | rows |
| 2 | Trial transitions | bins around each teleport (padded by `bins_before` before and `bins_after` + 400 ms settling after; defaults `bins_before = bins_after = 5` → ~2.6 s total) | rows |
| 3 | NaN positions | bins where `position == NaN` | rows |
| 4 | Low-performance trials | all bins of trials with `dPrime < threshDPrime` (defaults: `threshDPrime = 2` expert, `-1.1` naive) | rows |
| 5 | Recording boundaries | first `bins_before` (5) and last `bins_after` (5) bins of the whole recording | rows |
| 6 | Low-firing-rate neurons | neurons with mean firing rate below `firingMinimo` (default `0.1 Hz`) | columns |
| 7 | Unstable neurons | neurons whose firing rate drift index exceeds `threshDrift` across `numBlocks` halves of the recording (defaults: `threshDrift = 0.5`, `numBlocks = 2`) | columns |

## Time-bin filters (eliminate rows of X)

### 1. Immobility periods

**What it does**: bins inside a continuous stretch where
`velocity < thVel` for longer than `thDur` seconds are flagged for removal.
Each stretch is padded with `bins_before` bins after its end and
`bins_after` bins before its start (history-window compatibility — see
"How the filtering is applied mechanically" below).

**Why**: when the animal is stationary (whether licking water at the
reward, distracted at the start of a trial, or pausing mid-corridor), the
position signal is flat. Feeding such bins to a position decoder teaches
it to predict "animal is stationary at position X" — a high-confidence
claim about an intrinsically low-information sample. Removing them lets
the decoder train on intervals where position genuinely informs the
neural state.

**Note on running-drift**: when an animal continues running past the
declared corridor end (the rig extends the virtual corridor in
unrewarded trials), the encoder keeps registering position and the
animal is genuinely moving. Those bins carry real neural signal
correlated with locomotion. They are **not** filtered here — only
sustained immobility is.

**Parameters**:

| Parameter | Default | Meaning |
|---|---|---|
| `thVel` | 1 cm/s | velocity threshold |
| `thDur` | 4 s | minimum stretch duration |
| `bins_before` | 5 | history padding after the stretch ends |
| `bins_after` | 5 | history padding before the stretch starts |

`thVel = 1 cm/s` lands in the natural valley of the bimodal velocity
distribution (visualized in `exploration/01c-immobility-filter.py`):
one mode at 0 cm/s (stationary), another spread over 2–25 cm/s
(running). The threshold cleanly separates the two populations across
all datasets inspected.

`thDur = 4 s` is conservative: short pauses (< 4 s) are kept because
they are part of natural running behaviour. Only sustained immobility
is removed.

**Implementation**: `periods_immobility` in `analysis/preprocessing.py`,
a verbatim port from `ibioml.preprocess_data` (Sebas's
`feat/pre-processing` branch, commit `402af75`).

### 2. Trial transitions

Bins around each trial boundary are removed because the rig's teleport
creates an abrupt change in position and a brief settling period in the
neural activity.

**What it does**: at each `trialFinalBin`, the following bins are flagged:

- `bins_before` bins **before** the boundary (default 5 → ~1.0 s):
  history-window padding so no surviving sample has the teleport in its
  future feature window.
- The boundary itself (`trialFinalBin + 1`, first bin of the next trial).
- 2 bins of **settling** (400 ms post-teleport while the neural activity
  re-adapts to the new corridor input).
- `bins_after` bins **after** the settling (default 5 → ~1.0 s): history
  padding so no surviving sample has the teleport or settling in its
  past feature window.

With the defaults `bins_before = bins_after = 5` and `binLength = 200 ms`,
each boundary removes **13 bins (~2.6 s)** total. The 400 ms settling
window is set in the source as `int(400 / binLength)` — an experimental
choice based on the typical neural-adjustment latency to abrupt sensory
changes.

**Why**: at the moment of teleport, position jumps from the end of one
corridor to the start of the next in a single bin. The neural activity
around that teleport reflects adjustment / surprise, not
position-tracking. Without filtering, the decoder would see the
discontinuity as a contradiction in the data: a high position
immediately followed by position ≈ 0 with no movement signal in between.
That is an artefact of the rig's frame-of-reference, not the animal.

**Implementation**: the same `periods_immobility` function returns this
set as the `trial_transitions` list (alongside the `immobility` list).

### 3. NaN position bins

Bins where the position variable is `NaN` (a small number per recording,
typically isolated samples where the rig failed to register the wheel
encoder). Filtered as `np.where(np.isnan(y[:,0]))[0]`. No padding
applied — they are union-merged with the other indices and removed at
the end.

**Why**: a NaN target makes the sample untrainable for a regression
decoder; no point keeping it.

**Implementation**: inline in `preprocess_data` (Sebas's branch). Not
factored into its own function.

### 4. Low-performance trials

Trials whose `dPrime` falls below a threshold are removed in full —
every bin belonging to the trial is added to `indices_to_remove`.

**Thresholds** (from `external/scriptspipeline/all-datasets-experiment.py`):

- `threshDPrime = 2` for **expert** animals.
- `threshDPrime = -1.1` for **naive** animals.

`dPrime < threshold` flags trials where the animal had not yet learned
the task (or had momentarily stopped responding correctly). Including
them as training data would expose the decoder to neural states whose
behavioural label may not be reliable.

**Implementation**: `get_idx_by_low_performance` in
`ibioml.preprocess_data` (Sebas's branch, commit `402af75`).

### 5. Recording boundaries (first/last bins of the recording)

The first `bins_before` (default 5) and the last `bins_after` (default 5)
time bins of the entire recording are always removed. Their
`get_spikes_with_history` windows would partially fall outside the array
(filled with NaN or zeros), making them unfit for training.

**Implementation**: inline `first_indexes = np.arange(bins_before)` and
`last_indexes = np.arange(T - bins_after, T)` in `preprocess_data`. Note:
`periods_immobility` itself does **not** handle these — they are added
separately in the caller.

## Neuron filters (eliminate columns of X)

Two filters operate on the neuron axis after the row filtering above is
complete. They are computed from the cleaned `neural_data` (i.e. with
immobility, NaN, etc. already removed) so the firing-rate estimates
reflect only valid navigation periods.

### 6. Low-firing-rate neurons

Neurons whose mean firing rate over the surviving bins falls below
`firingMinimo` are removed.

**Threshold**: `firingMinimo = 0.1 Hz` (default). Very low — only
nearly-silent units are eliminated. The mean is computed as
`np.nanmean(neural_data, axis=0) / (binLength / 1000)`.

**Why**: a neuron with almost no spikes contributes no usable signal
and adds dimensionality cost to the feature vector.

**Implementation**: `clean_neurons_by_low_firing_rate` in
`ibioml.preprocess_data`.

### 7. Unstable neurons (firing-rate drift)

Neurons whose firing rate **drifts across the recording** are removed.
This is different from the "position drift" the corridor produces;
here "drift" means the neuron itself changes its baseline activity
over the recording session (electrode movement, neuron health, etc.).

**Method**: split the cleaned recording into `numBlocks` equal blocks
(default `numBlocks = 2`, i.e. halves), compute mean firing rate per
neuron per block, and define a drift index per neuron:

`driftIndex[n] = |mean_block_1[n] - mean_block_2[n]| / (mean_block_1[n] + mean_block_2[n])`

Neurons with `driftIndex > threshDrift` (default `threshDrift = 0.5`)
are removed.

**Why**: an unstable neuron is a moving target — its average response
during early training trials would differ systematically from its
response during later trials, biasing the decoder.

**Implementation**: `clean_unstable_neurons` in
`ibioml.preprocess_data`. New addition by Sebas in `feat/pre-processing`
(not present in `main`).

## How the filtering is applied mechanically

Three observations about the order of operations:

**1. History windows are applied BEFORE any time-bin filtering.**

```python
X = get_spikes_with_history(neural_data_with_ctxt, bins_before, ...)  # T_bins → T_bins of feature vectors
X = np.delete(X, indices_to_remove, 0)                                 # then drop bad rows
```

This is deliberate. If filtering happened first, the array would no
longer be temporally contiguous and the history function would
mistakenly assemble feature vectors that mix bins from different
trials (the gap between filtered-out bins would be invisible to it).
By applying history first, each surviving row carries a faithful
temporal window of bins; the row is dropped iff its window touches a
flagged bin — which is exactly why every filter includes a ±5-bin
padding for history-window safety.

**2. Time-bin filtering and neuron filtering happen in different passes.**

Time-bin filtering uses `np.delete(X, indices_to_remove, axis=0)` to
drop full rows of `X`. Neuron filtering uses
`np.delete(X, neurons_to_remove, axis=2)` to drop full columns of the
feature dimension. The two are independent.

**3. The same row indices are deleted from `y` and `trial_markers`.**

```python
X             = np.delete(X,             indices_to_remove, 0)
y             = np.delete(y,             indices_to_remove, 0)
trial_markers = np.delete(trial_markers, indices_to_remove, 0)
```

This keeps input, target, and trial-id arrays perfectly synchronised.

## Parameters summary

All thresholds sourced from
`external/scriptspipeline/all-datasets-experiment.py`:

| Parameter | Value | Filter | Meaning |
|---|---|---|---|
| `thVel` | 1 cm/s | immobility | velocity threshold |
| `thDur` | 4 s | immobility | minimum stretch duration |
| `bins_before` | 5 | history padding | bins back in history window |
| `bins_after` | 5 | history padding | bins forward in history window |
| `bins_current` | 1 | history padding | current bin (always 1) |
| `binLength` | 200 ms | all | bin duration of the .mat |
| `threshDPrime` | 2 / -1.1 | low performance | expert / naive |
| `firingMinimo` | 0.1 Hz | low firing | minimum mean firing rate |
| `numBlocks` | 2 | unstable neurons | halves the recording |
| `threshDrift` | 0.5 | unstable neurons | drift index threshold |

## Validation: cross-dataset behavior

Applied to the four sample expert datasets covering both regions, the
fraction of bins removed by the **immobility + trial-transition** filters
is consistent in magnitude (36–49 %) and varies with how often the animal
stops, not with the recording region:

| Dataset | total bins | % immobility | % trial trans | % union |
|---|---|---|---|---|
| `Pir_L5_expert_bins200ms` | 19564 | 37.9 % | 11.7 % | 48.9 % |
| `Pir_L3_expert_bins200ms` | 15493 | 17.6 % | 20.9 % | 36.5 % |
| `CA3_S18_expert_bins200ms` | 20518 | 27.9 % | 14.9 % | 40.7 % |
| `DG_S19_expert_bins200ms` | 18504 | 23.1 % | 15.7 % | 37.8 % |

Trial-transition coverage is near-constant (12–21 %) because it depends
only on the number of trials and the padding. Immobility coverage
varies with the animal's behaviour: `Pir_L5` is the outlier (an animal
that includes a single ~380 s static trial drives much of that
37.9 %).

The filter behaves as intended in two further respects, also verified
during validation (see plots `f3-filter-on-v4b.png` across datasets):

- It captures the **lick-drift** bins — they show up either as
  immobility bins or as trial-transition bins, depending on how close
  to the trial boundary the licking sits.
- It does **not** remove the running-drift bins (velocity 10–25 cm/s,
  position past 60 cm), consistent with the design rationale (running
  bins carry real neural signal correlated with locomotion).

The other filters (NaN, low-performance, recording boundaries,
low-firing, unstable neurons) have not yet been validated independently
in this project — they are inherited from Sebas's pipeline with their
parameter defaults and will be exercised once `analysis/preprocessing.py`
extends beyond `periods_immobility`.

## Open / deferred

- **Conversion factor `position → cm`**. Pending confirmation with the
  lab (see `docs/trial-structure.md` open questions). Independent of all
  filters above (they operate on velocity in cm/s and on bin / neuron
  indices).
- **Migration to `ibioml.preprocess_data`**. The full pipeline currently
  lives in Sebas's `feat/pre-processing` branch upstream. Once that
  branch merges to `main`, `analysis/preprocessing.py` should be deleted
  and the imports switched to `from ibioml.preprocess_data import …`.
- **Implementation gap in `analysis/preprocessing.py`**. Today this
  project only ports `periods_immobility`. Filters 3–7 (NaN, low-perf,
  boundaries, low-firing, unstable) are documented here for completeness
  but live exclusively in `ibioml.preprocess_data` until the migration.
- **History windows** (`get_spikes_with_history`): documented here only
  as far as filter-pipeline mechanics; full description in
  `ibioml.utils.preprocessing_funcs.get_spikes_with_history`.

## Code references

- `analysis/preprocessing.py` — local port of `periods_immobility`.
- `exploration/01c-immobility-filter.py` — validation notebook for the
  immobility + trial-transition filter (plots referenced above).
- `external/scriptspipeline/all-datasets-experiment.py` — Sebas's
  reference values for the thresholds.
- Upstream source: `feat/pre-processing` branch of
  `mariburginlab-labPrograms/IBioML`, commit `402af75`,
  `ibioml/preprocess_data.py` (full pipeline).
