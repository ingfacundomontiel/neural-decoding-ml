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

The 21-file mapping (canonical lab filename → destination in this repo):

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
| `L7_naive_bins200ms_completo.mat`         | `piriform/Pir_L7_naive_bins200ms.mat`         |
| `N53_naive_bins200ms_completo.mat`        | `piriform/Pir_N53_naive_bins200ms.mat`        |
| `CA3_S17_bins200ms_completo.mat`          | `ca3/CA3_S17_expert_bins200ms.mat`            |
| `CA3_S18_bins200ms_completo.mat`          | `ca3/CA3_S18_expert_bins200ms.mat`            |
| `CA3_S30_bins200ms_completo.mat`          | `ca3/CA3_S30_naive_bins200ms.mat`             |
| `CA3_S31_bins200ms_completo.mat`          | `ca3/CA3_S31_naive_bins200ms.mat`             |
| `DG_S17_bins200ms_completo.mat`           | `dg/DG_S17_expert_bins200ms.mat`              |
| `DG_S18_bins200ms_completo.mat`           | `dg/DG_S18_expert_bins200ms.mat`              |
| `DG_S19_bins200ms_completo.mat`           | `dg/DG_S19_expert_bins200ms.mat`              |
| `DG_S24_bins200ms_completo.mat`           | `dg/DG_S24_naive_bins200ms.mat`               |
| `DG_S38_bins200ms_completo.mat`           | `dg/DG_S38_naive_bins200ms.mat`               |
| `DG_S39_bins200ms_completo.mat`           | `dg/DG_S39_naive_bins200ms.mat`               |
| `DG_S41_bins200ms_completo.mat`           | `dg/DG_S41_naive_bins200ms.mat`               |

The lab sources may live in any of `sfn/datasets/`, `san/datasets/`, or
`datasets/` under the experiments root — for byte-identical duplicates,
`sfn/datasets/` is the preferred source (per D-010). The complete bundle
that the lab maintainer distributes (`scriptspipeline/datasets.zip`) contains exactly
the 21 files listed above with the `_completo` suffix.

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
`ibioml.preprocess_data` (history windows, trial filtering by dPrime,
immobility filtering, dropping low-firing neurons, etc.) and produces the
artifacts under `data/processed/`.

When you write the Methods section of the thesis, distinguish:
1. What was done upstream (before `.mat`) — described above.
2. What `ibioml.preprocess_data` does (documented in
   [`docs/ibioml-audit.md`](../../docs/ibioml-audit.md) §1).
3. What experiment-specific code adds on top of (2).

## Naming convention

```
<region>_<animal_id>_<cohort>_bins<size>.mat
```

- **`<region>`** — `Pir`, `CA3`, or `DG`. Redundant with the parent directory
  but kept in the filename so individual files remain self-describing if they
  get moved or copied out of context.
- **`<animal_id>`** — e.g. `L3`, `L5`, `N31`, `N53`, `L7`, `S17`, `S18`, …
- **`<cohort>`** — `expert` or `naive` (learning stage).
- **`bins<size>`** — `bins100ms` or `bins200ms` (temporal bin width).

The `_completo` suffix of the lab's filenames is dropped during the rename
(it conveyed no useful information not already in the path or the cohort
field).

## File format

Every `.mat` exposes the same MATLAB variables (semantics in
[`specs/research.md`](../../specs/research.md) § Common data model):

| Variable | Shape | Meaning |
|---|---|---|
| `neuronActivity` | `(T_bins, N_neurons)` | spike counts per bin per neuron |
| `position` | `(T_bins, 1)` | animal's VR-corridor position (wheel-encoder degrees; see [`docs/trial-structure.md`](../../docs/trial-structure.md)) |
| `velocity` or `speed` | `(T_bins, 1)` | velocity in cm/s (name inconsistent across datasets) |
| `rewCtxt` | `(T_bins,)` | per-trial label, encoded per-bin (constant within trial): 1 if the trial's context is the rewarded one |
| `rewOdor` | `(T_bins,)` | per-trial label, encoded per-bin (constant within trial): 1 if the trial's odor is the rewarded one |
| `goResponse` | `(T_bins,)` | per-trial label, encoded per-bin (constant within trial): 1 if the animal licked at any point during the trial |
| `dPrime` | `(E_trials,)` | cumulative discriminability at that trial |
| `criterion` | `(E_trials,)` | cumulative licking bias |
| `trialDurationInBins` | `(E_trials,)` | length of each trial |
| `trialFinalBin` | `(E_trials,)` | last temporal bin of each trial (1-indexed in MATLAB; load with `-1`) |

> Important: `rewCtxt`, `rewOdor`, and `goResponse` are stored as
> per-bin arrays but are **per-trial labels** — their value is constant
> within each trial. Verified empirically across 4 datasets (886 trials,
> zero exceptions). You cannot use them to locate *when* within a trial
> the animal licked, the odor was presented, or the context became
> rewarded; only *which* trials had those properties.
>
> To extract a clean per-trial array, sample at any bin inside the
> trial (e.g. `value[trialFinalBin]`). To detect the *moment* of
> licking within a trial — for instance to filter the bins where the
> animal is licking the reward — use `velocity` instead (it drops to
> ~0 cm/s during licking).

## Catalog

### Piriform cortex — `piriform/` (10 files, 6 animals)

| Filename | Animal | Cohort | Bin | In SFN | Notes |
|---|---|---|---|---|---|
| `Pir_L3_expert_bins100ms.mat`   | L3  | expert | 100 ms | ✓ | |
| `Pir_L3_expert_bins200ms.mat`   | L3  | expert | 200 ms | ✓ | |
| `Pir_L5_expert_bins100ms.mat`   | L5  | expert | 100 ms | ✓ | |
| `Pir_L5_expert_bins200ms.mat`   | L5  | expert | 200 ms | ✓ | |
| `Pir_N31_expert_bins100ms.mat`  | N31 | expert | 100 ms | ✓ | |
| `Pir_N31_expert_bins200ms.mat`  | N31 | expert | 200 ms | ✓ | |
| `Pir_N59_expert_bins100ms.mat`  | N59 | expert | 100 ms | ✗ | **not in SFN poster** — confirm inclusion with director |
| `Pir_N59_expert_bins200ms.mat`  | N59 | expert | 200 ms | ✗ | **not in SFN poster** — confirm inclusion with director |
| `Pir_L7_naive_bins200ms.mat`    | L7  | naive  | 200 ms | ✓ | byte-identical to lab's `L7_naive_bins200ms_completo.mat` |
| `Pir_N53_naive_bins200ms.mat`   | N53 | naive  | 200 ms | ✓ | byte-identical to lab's `N53_naive_bins200ms_completo.mat` |

### CA3 — `ca3/` (4 files, 4 animals)

| Filename | Animal | Cohort | Bin | In SFN | Notes |
|---|---|---|---|---|---|
| `CA3_S17_expert_bins200ms.mat` | S17 | expert | 200 ms | ✓ | animal also recorded in DG (`DG_S17`, separate recording) |
| `CA3_S18_expert_bins200ms.mat` | S18 | expert | 200 ms | ✓ | animal also recorded in DG (`DG_S18`, separate recording) |
| `CA3_S30_naive_bins200ms.mat`  | S30 | naive  | 200 ms | ✓ | |
| `CA3_S31_naive_bins200ms.mat`  | S31 | naive  | 200 ms | ✓ | |

### Dentate gyrus — `dg/` (7 files, 7 animals)

| Filename | Animal | Cohort | Bin | In SFN | Notes |
|---|---|---|---|---|---|
| `DG_S17_expert_bins200ms.mat` | S17 | expert | 200 ms | ✓ | animal also recorded in CA3 (`CA3_S17`, separate recording) |
| `DG_S18_expert_bins200ms.mat` | S18 | expert | 200 ms | ✓ | animal also recorded in CA3 (`CA3_S18`, separate recording) |
| `DG_S19_expert_bins200ms.mat` | S19 | expert | 200 ms | ✓ | |
| `DG_S24_naive_bins200ms.mat`  | S24 | naive  | 200 ms | ✗ | **not in SFN poster** — confirm inclusion with director |
| `DG_S38_naive_bins200ms.mat`  | S38 | naive  | 200 ms | ✓ | |
| `DG_S39_naive_bins200ms.mat`  | S39 | naive  | 200 ms | ✓ | |
| `DG_S41_naive_bins200ms.mat`  | S41 | naive  | 200 ms | ✗ | **not in SFN poster** — confirm inclusion with director |

## Cross-reference with the SFN poster

The SFN poster groups datasets as six cohorts (region × learning stage):

| Poster group | Expected | Present in `data/raw/` | Status |
|---|---|---|---|
| Pir-Naive   | N53, L7         | N53, L7                       | complete |
| Pir-Expert  | N31, L3, L5     | N31, L3, L5 (+ N59 extra)     | complete + 1 extra |
| CA3-Naive   | S30, S31        | S30, S31                      | complete |
| CA3-Expert  | S17, S18        | S17, S18                      | complete |
| DG-Naive    | S38, S39        | S38, S39 (+ S24, S41 extras)  | complete + 2 extras |
| DG-Expert   | S17, S18, S19   | S17, S18, S19                 | complete |

Animals S17 and S18 were each **recorded twice — once in CA3 and once in
DG, not simultaneously**. The recurring labels in the poster reflect the
same animal recorded in different regions, not duplicate files. The
identifier (`S17`, `L5`, `N53`, etc.) refers to the animal across all its
recordings, regardless of region or learning stage.

**Extras on disk** (in `data/raw/`, not shown in poster): `Pir_N59`
(100 ms + 200 ms), `DG_S24`, `DG_S41`. Likely datasets the director chose
not to include in the SFN visualization but that may still be valid for the
thesis — confirm with director whether to keep, exclude, or treat
separately.

## Archived `_original` snapshots

The lab's `L7` and `N53` files used to ship in two flavours: a `_completo`
version and an `_original` snapshot, with substantially different sizes.
An MD5 check (2026-06-19) confirmed that **the `_completo` version is the
one we have here** as `Pir_L7_naive_bins200ms.mat` / `Pir_N53_naive_bins200ms.mat`
(byte-identical to the lab bundle).

The `_original` files are an older pre-processing snapshot that the lab no
longer ships in its production pipeline. They are kept on the developer
machine only, under `_legacy/datasets/originals/`, gitignored — useful as
a reference if a comparison vs the production version is ever needed, but
not part of the canonical catalog.

## Open questions

1. **`velocity` vs `speed`.** Variable naming is inconsistent across
   datasets. The first overview notebook should produce a column listing
   which datasets use which name.

## Do not modify

These are the canonical source-of-truth files for every derived artifact in
the project. If one needs updating (new recording session, corrected data),
replace the whole file, bump a version note in the relevant row above, and
rerun every dependent preprocessing step.
