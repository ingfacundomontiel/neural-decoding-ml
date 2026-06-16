# `data/raw/` — Dataset Catalog

All `.mat` files used as input to the project's preprocessing pipeline. One
canonical copy per dataset, deduplicated from the three source directories per
[`specs/dataset-inventory.md`](../../specs/dataset-inventory.md).

Files are organized into subdirectories by **recording region**
(`piriform/`, `ca3/`, `dg/`). The cohort (`expert` / `naive`) is embedded in
the filename. See [Naming convention](#naming-convention) below.

> **For the experimental setup and trial structure** (corridor geometry,
> reward zone, position units, known drift), see
> [`docs/trial-structure.md`](../../docs/trial-structure.md).

## Obtaining the files on a new machine

The `.mat` files are **not tracked in git** — see
[`specs/decisions.md`](../../specs/decisions.md) D-013 for the rationale
(unpublished lab data, not ours to redistribute through GitHub). To populate
`data/raw/` on a fresh clone, copy from the lab source paths documented in
[`specs/dataset-inventory.md`](../../specs/dataset-inventory.md), applying
the renames below.

Lab source root (on the IBioBA machine):
```
/Users/.../marinburginlab-labPrograms/neural-decoding/notebooks/experiments/
```

The 20-file mapping (canonical lab filename → destination in this repo):

| Lab source filename                       | Destination under `data/raw/` |
|---|---|
| `L3_bins100ms_completo.mat`               | `piriform/Pir_L3_expert_bins100ms.mat`        |
| `L3_bins200ms_completo.mat`               | `piriform/Pir_L3_expert_bins200ms.mat`        |
| `L5_bins100ms_completo.mat`               | `piriform/Pir_L5_expert_bins100ms.mat`        |
| `L5_bins200ms_completo.mat`               | `piriform/Pir_L5_expert_bins200ms.mat`        |
| `N31_bins100ms_completo.mat`              | `piriform/Pir_N31_expert_bins100ms.mat`       |
| `N31_bins200ms_completo.mat`              | `piriform/Pir_N31_expert_bins200ms.mat`       |
| `N59_bins100ms_completo.mat`              | `piriform/Pir_N59_expert_bins100ms.mat`       |
| `N59_bins200ms_completo.mat`              | `piriform/Pir_N59_expert_bins200ms.mat`       |
| `L7_naive_bins200ms.mat`                  | `piriform/Pir_L7_naive_bins200ms.mat`         |
| `L7_naive_bins200ms_original.mat`         | `piriform/Pir_L7_naive_bins200ms_original.mat`|
| `N53_naive_bins200ms.mat`                 | `piriform/Pir_N53_naive_bins200ms.mat`        |
| `N53_naive_bins200ms_original.mat`        | `piriform/Pir_N53_naive_bins200ms_original.mat`|
| `CA3_S18_bins200ms_completo.mat`          | `ca3/CA3_S18_expert_bins200ms.mat`            |
| `CA3_S30_bins200ms_completo.mat`          | `ca3/CA3_S30_naive_bins200ms.mat`             |
| `CA3_S31_bins200ms_completo.mat`          | `ca3/CA3_S31_naive_bins200ms.mat`             |
| `DG_S19_bins200ms_completo.mat`           | `dg/DG_S19_expert_bins200ms.mat`              |
| `DG_S24_bins200ms_completo.mat`           | `dg/DG_S24_naive_bins200ms.mat`               |
| `DG_S38_bins200ms_completo.mat`           | `dg/DG_S38_naive_bins200ms.mat`               |
| `DG_S39_bins200ms_completo.mat`           | `dg/DG_S39_naive_bins200ms.mat`               |
| `DG_S41_bins200ms_completo.mat`           | `dg/DG_S41_naive_bins200ms.mat`               |

The lab sources may live in any of `sfn/datasets/`, `san/datasets/`, or
`datasets/` under the experiments root — for byte-identical duplicates,
`sfn/datasets/` is the preferred source (per D-010).

## What "raw" means here

These files are **not raw recordings**. They have already been processed
upstream (before they reached this repo):

- **Binning is done.** `neuronActivity` is `(T_bins, N_neurons)` with spike
  counts already aggregated in 100 ms or 200 ms bins.
- **Spike sorting is done.** The `N_neurons` axis represents sorted units.
- **Trials are structured.** `trialFinalBin`, `trialDurationInBins`,
  `dPrime`, `criterion` are precomputed.
- **Behavior is aligned to bins.** `position` and `velocity`/`speed` are
  resampled to the same temporal grid as the neural activity.

What **this project** still does on top of these files lives in
`ibioml.preprocess_data` (history windows, trial filtering by dPrime, dropping
low-firing neurons, etc.) and produces the artifacts under `data/processed/`.

When you write the Methods section of the thesis, distinguish:
1. What was done upstream (before `.mat`) — described above.
2. What `ibioml.preprocess_data` does (documented in
   [`docs/ibioml-audit.md`](../../docs/ibioml-audit.md) §1).
3. What experiment-specific code adds on top of (2).

## Naming convention

```
<region>_<animal_id>_<cohort>_bins<size>[_suffix].mat
```

- **`<region>`** — `Pir`, `CA3`, or `DG`. Redundant with the parent directory
  but kept in the filename so individual files remain self-describing if they
  get moved or copied out of context.
- **`<animal_id>`** — e.g. `L3`, `L5`, `N31`, `N53`, `L7`, `S18`, `S19`, …
- **`<cohort>`** — `expert` or `naive` (learning stage).
- **`bins<size>`** — `bins100ms` or `bins200ms` (temporal bin width).
- **`[_suffix]`** — optional, e.g. `_original` for the naive piriform pre-/post-
  processing pair (see Open questions).

## File format

Every `.mat` exposes the same MATLAB variables (semantics in
[`specs/research.md`](../../specs/research.md) § Common data model):

| Variable | Shape | Meaning |
|---|---|---|
| `neuronActivity` | `(T_bins, N_neurons)` | spike counts per bin per neuron |
| `position` | `(T_bins, 1)` | animal's VR-corridor position |
| `velocity` or `speed` | `(T_bins, 1)` | velocity (name inconsistent across datasets) |
| `rewCtxt` | `(E_trials,)` | 1 if trial's context is rewarded |
| `rewOdor` | `(E_trials,)` | 1 if trial's odor is rewarded |
| `goResponse` | `(E_trials,)` | 1 if mouse licked |
| `dPrime` | `(E_trials,)` | cumulative discriminability at that trial |
| `criterion` | `(E_trials,)` | cumulative licking bias |
| `trialDurationInBins` | `(E_trials,)` | length of each trial |
| `trialFinalBin` | `(E_trials,)` | last temporal bin of each trial |

## Catalog

### Piriform cortex — `piriform/` (12 files, 6 animals)

| Filename | Animal | Cohort | Bin | In SFN | Notes |
|---|---|---|---|---|---|
| `Pir_L3_expert_bins100ms.mat`        | L3  | expert | 100 ms | ✓ | |
| `Pir_L3_expert_bins200ms.mat`        | L3  | expert | 200 ms | ✓ | |
| `Pir_L5_expert_bins100ms.mat`        | L5  | expert | 100 ms | ✓ | |
| `Pir_L5_expert_bins200ms.mat`        | L5  | expert | 200 ms | ✓ | |
| `Pir_N31_expert_bins100ms.mat`       | N31 | expert | 100 ms | ✓ | |
| `Pir_N31_expert_bins200ms.mat`       | N31 | expert | 200 ms | ✓ | |
| `Pir_N59_expert_bins100ms.mat`       | N59 | expert | 100 ms | ✗ | **not in SFN poster** — confirm inclusion with director |
| `Pir_N59_expert_bins200ms.mat`       | N59 | expert | 200 ms | ✗ | **not in SFN poster** — confirm inclusion with director |
| `Pir_L7_naive_bins200ms.mat`             | L7  | naive  | 200 ms | ✓ | processed version (see Open questions) |
| `Pir_L7_naive_bins200ms_original.mat`    | L7  | naive  | 200 ms | ✓ | `_original` snapshot (smaller; see Open questions) |
| `Pir_N53_naive_bins200ms.mat`            | N53 | naive  | 200 ms | ✓ | processed version (see Open questions) |
| `Pir_N53_naive_bins200ms_original.mat`   | N53 | naive  | 200 ms | ✓ | `_original` snapshot (smaller; see Open questions) |

### CA3 — `ca3/` (3 files, 3 sessions)

| Filename | Animal/Session | Cohort | Bin | In SFN | Notes |
|---|---|---|---|---|---|
| `CA3_S18_expert_bins200ms.mat` | S18 | expert | 200 ms | ✓ | same animal as `DG_S18` (different recording session) |
| `CA3_S30_naive_bins200ms.mat`  | S30 | naive  | 200 ms | ✓ | |
| `CA3_S31_naive_bins200ms.mat`  | S31 | naive  | 200 ms | ✓ | |

### Dentate gyrus — `dg/` (5 files, 5 sessions)

| Filename | Animal/Session | Cohort | Bin | In SFN | Notes |
|---|---|---|---|---|---|
| `DG_S19_expert_bins200ms.mat` | S19 | expert | 200 ms | ✓ | |
| `DG_S24_naive_bins200ms.mat`  | S24 | naive  | 200 ms | ✗ | **not in SFN poster** — confirm inclusion with director |
| `DG_S38_naive_bins200ms.mat`  | S38 | naive  | 200 ms | ✓ | |
| `DG_S39_naive_bins200ms.mat`  | S39 | naive  | 200 ms | ✓ | |
| `DG_S41_naive_bins200ms.mat`  | S41 | naive  | 200 ms | ✗ | **not in SFN poster** — confirm inclusion with director |

## Cross-reference with the SFN poster

The SFN poster groups datasets as six cohorts (region × learning stage):

| Poster group | Expected in poster | Present in `data/raw/` | Missing |
|---|---|---|---|
| Pir-Naive   | N53, L7         | N53, L7                       | — |
| Pir-Expert  | N31, L3, L5     | N31, L3, L5 (+ N59 extra)     | — |
| CA3-Naive   | S30, S31        | S30, S31                      | — |
| CA3-Expert  | S17, S18        | S18                           | **S17** |
| DG-Naive    | S38, S39        | S38, S39 (+ S24, S41 extras)  | — |
| DG-Expert   | S17, S18, S19   | S19                           | **S17, S18** |

Animals S17 and S18 contributed **two separate recording sessions each** —
one in CA3 and one in DG (not simultaneous). The recurring labels in the
poster reflect the same animal, not duplicate files.

**Missing from disk** (in poster, not yet in `data/raw/`):
`CA3_S17`, `DG_S17`, `DG_S18`. To be obtained from the director.

**Extras on disk** (in `data/raw/`, not shown in poster): `Pir_N59`
(100 ms + 200 ms), `DG_S24`, `DG_S41`. Likely datasets the director chose
not to include in the SFN visualization but that may still be valid for the
thesis — confirm whether to keep, exclude, or treat separately.

## Open questions

1. **`_original` vs non-`_original` naive piriform files.** L7 and N53 each
   have two versions, with the non-`_original` ones being substantially
   larger (~3–6× by bytes). It is not yet established what processing step
   produced each variant. To be resolved by a comparison in
   `exploration/`, not by file inspection alone.
2. **What `_completo` meant in the original filenames.** Removed during the
   rename, but the original convention may carry information worth surfacing
   in the manifest (e.g. "all trials, no filtering" vs a curated subset).
3. **`velocity` vs `speed`.** Variable naming is inconsistent across datasets.
   The first overview notebook should produce a column listing which
   datasets use which name.
4. **Animal vs session ID.** Hippocampus datasets are labeled by session
   (`S17`, `S19`, …); piriform datasets by animal (`L3`, `N31`, …). The
   lab's master record likely clarifies how to relate the two.

## Do not modify

These are the canonical source-of-truth files for every derived artifact in
the project. If one needs updating (new recording session, corrected data),
replace the whole file, bump a version note in the relevant row above, and
rerun every dependent preprocessing step.
