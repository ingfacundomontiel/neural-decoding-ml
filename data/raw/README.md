# `data/raw/` — Dataset Catalog

All 20 raw `.mat` files used across the project. One canonical copy per dataset,
deduplicated from the three source directories per
[`specs/dataset-inventory.md`](../../specs/dataset-inventory.md).

## File format

Every `.mat` exposes the same MATLAB variables (semantics documented in
`specs/research.md § Common data model`):

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

### Hippocampus — 200 ms bins (8 datasets)

| Filename | Size (bytes) | Region | Animal | Session | Notes |
|---|---|---|---|---|---|
| `CA3_S18_bins200ms_completo.mat` | 151,334 | CA3 | — | S18 | |
| `CA3_S30_bins200ms_completo.mat` | 74,065 | CA3 | — | S30 | new in SFN batch |
| `CA3_S31_bins200ms_completo.mat` | 500,432 | CA3 | — | S31 | new in SFN batch |
| `DG_S19_bins200ms_completo.mat` | 129,192 | DG | — | S19 | |
| `DG_S24_bins200ms_completo.mat` | 719,632 | DG | — | S24 | new in SFN batch |
| `DG_S38_bins200ms_completo.mat` | 1,510,720 | DG | — | S38 | largest file |
| `DG_S39_bins200ms_completo.mat` | 529,248 | DG | — | S39 | new in SFN batch |
| `DG_S41_bins200ms_completo.mat` | 483,880 | DG | — | S41 | new in SFN batch |

### Expert piriform cortex — 200 ms bins (4 datasets)

| Filename | Size (bytes) | Region | Animal | Cohort |
|---|---|---|---|---|
| `L3_bins200ms_completo.mat` | 146,583 | piriform | L3 | expert |
| `L5_bins200ms_completo.mat` | 302,264 | piriform | L5 | expert |
| `N31_bins200ms_completo.mat` | 198,330 | piriform | N31 | expert |
| `N59_bins200ms_completo.mat` | 122,371 | piriform | N59 | expert |

### Expert piriform cortex — 100 ms bins (4 datasets)

| Filename | Size (bytes) | Region | Animal | Cohort | Notes |
|---|---|---|---|---|---|
| `L3_bins100ms_completo.mat` | 239,379 | piriform | L3 | expert | only in legacy dirs |
| `L5_bins100ms_completo.mat` | 449,705 | piriform | L5 | expert | |
| `N31_bins100ms_completo.mat` | 302,545 | piriform | N31 | expert | only in legacy dirs |
| `N59_bins100ms_completo.mat` | 195,212 | piriform | N59 | expert | only in legacy dirs |

### Naive piriform cortex — 200 ms bins (4 files, 2 animals)

| Filename | Size (bytes) | Region | Animal | Cohort | Notes |
|---|---|---|---|---|---|
| `L7_naive_bins200ms.mat` | 1,333,264 | piriform | L7 | naive | processed version |
| `L7_naive_bins200ms_original.mat` | 390,746 | piriform | L7 | naive | pre-processing snapshot |
| `N53_naive_bins200ms.mat` | 1,471,696 | piriform | N53 | naive | processed version |
| `N53_naive_bins200ms_original.mat` | 256,986 | piriform | N53 | naive | pre-processing snapshot |

## Cohort groupings (summary)

- **Expert piriform:** L3, L5, N31, N59 (4 animals; both 100 ms and 200 ms bins available)
- **Naive piriform:** L7, N53 (2 animals; 200 ms bins, `_original` and processed versions)
- **Hippocampus:** CA3_S18, CA3_S30, CA3_S31, DG_S19, DG_S24, DG_S38, DG_S39, DG_S41
  (8 sessions; 200 ms bins only)

## Notes and open questions

1. **`_original` naive files.** For L7 and N53 there are both `_original` and
   non-`_original` versions with substantially different sizes. Need to confirm
   with Facundo what the processing step was between them. One hypothesis: the
   non-`_original` files have had trials pre-cleaned.
2. **`velocity` vs `speed`.** Variable naming is inconsistent across datasets.
   First-pass exploration should confirm which datasets use which name.
3. **Animal vs session metadata.** The filename embeds animal (L3, L5, ...) or
   session (S18, S19, ...) but not both. The lab's master record likely clarifies;
   ask Facundo to fill in the "Animal" column for hippocampus entries and the
   "Session" column for piriform entries if relevant.
4. **Sizes are not strictly proportional to neuron/trial counts.** A reliable
   count requires loading each file (done in `exploration/01-dataset-overview.py`).

## Do not modify

These are the canonical source-of-truth files for every derived artifact in the
project. If one needs updating (new recording session, corrected data), replace
the whole file, bump a version note in the relevant row above, and rerun every
dependent preprocessing step.
