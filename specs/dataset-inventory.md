# Dataset Inventory

Complete catalog of every `.mat` file found across the three source directories,
with deduplication decisions for migration to `data/raw/`.

Generated: 2026-04-17. Based on byte-identical comparison via file size + mtime.

## Legend

- `sfn` = `marinburginlab-labPrograms/neural-decoding/notebooks/experiments/sfn/datasets/`
- `san` = `marinburginlab-labPrograms/neural-decoding/notebooks/experiments/san/datasets/`
- `mlp-exp` = `marinburginlab-labPrograms/neural-decoding/notebooks/experiments/datasets/`
- `nd-ml` = `neural-decoding-ml/datasets/`
- `nd` = `neural-decoding/datasets/`

## Summary

- **Authoritative source per file:** `sfn/` (Oct 31 2025) wherever available,
  falling back to older dirs only for files `sfn/` doesn't have.
- **Byte-identical copies across directories:** confirmed by identical file sizes;
  older copies (2024-10-13, 2025-02-26) match 2025-10-11 and 2025-10-31 copies exactly
  — they're the same data re-timestamped by `cp`.
- **Unique files only in older dirs:** the 100ms expert/naive piriform datasets
  (L3, L5, N31, N59 in 100ms bins), plus some legacy-named files (`L3.mat`, `L5.mat`,
  `n31.mat`, `N59.mat`, `L5_200msbin.mat`).
- **Unique files only in `sfn/`:** 5 new hippocampus datasets (CA3_S30, CA3_S31,
  DG_S24, DG_S38, DG_S39, DG_S41).

## Authoritative catalog — files going to `data/raw/`

### Hippocampus (200ms bins) — all from `sfn/`

| Filename | Size (bytes) | Notes |
|---|---|---|
| `CA3_S18_bins200ms_completo.mat` | 151,334 | |
| `CA3_S30_bins200ms_completo.mat` | 74,065 | new in sfn |
| `CA3_S31_bins200ms_completo.mat` | 500,432 | new in sfn |
| `DG_S19_bins200ms_completo.mat` | 129,192 | |
| `DG_S24_bins200ms_completo.mat` | 719,632 | new in sfn |
| `DG_S38_bins200ms_completo.mat` | 1,510,720 | new in sfn |
| `DG_S39_bins200ms_completo.mat` | 529,248 | new in sfn |
| `DG_S41_bins200ms_completo.mat` | 483,880 | new in sfn |

### Expert piriform (200ms bins) — all from `sfn/`

| Filename | Size (bytes) | Notes |
|---|---|---|
| `L3_bins200ms_completo.mat` | 146,583 | |
| `L5_bins200ms_completo.mat` | 302,264 | |
| `N31_bins200ms_completo.mat` | 198,330 | |
| `N59_bins200ms_completo.mat` | 122,371 | |

### Expert piriform (100ms bins) — only in older dirs, take from `nd/` or `nd-ml/`

| Filename | Size (bytes) | Source | Notes |
|---|---|---|---|
| `L3_bins100ms_completo.mat` | 239,379 | nd/nd-ml | not in marinburginlab |
| `L5_bins100ms_completo.mat` | 449,705 | sfn (also older) | take from sfn |
| `N31_bins100ms_completo.mat` | 302,545 | nd/nd-ml | not in marinburginlab |
| `N59_bins100ms_completo.mat` | 195,212 | nd/nd-ml | not in marinburginlab |

### Naive piriform (200ms bins) — all from `sfn/`

| Filename | Size (bytes) | Notes |
|---|---|---|
| `L7_naive_bins200ms.mat` | 1,333,264 | processed version |
| `L7_naive_bins200ms_original.mat` | 390,746 | original — **keep both** |
| `N53_naive_bins200ms.mat` | 1,471,696 | processed version |
| `N53_naive_bins200ms_original.mat` | 256,986 | original — **keep both** |

## Files being dropped (duplicates or legacy)

### Byte-identical duplicates (safe to drop — same data)

- `mlp-exp/` copies of the 11 files that also exist in `sfn/` (San Fran dataset was
  copied to a shared `experiments/datasets/` folder before `sfn/` was created).
- `san/datasets/` copies of the 11 files — same as above.
- `nd/`, `nd-ml/` copies of 200ms expert/hippocampus files that are already in `sfn/`.

### Legacy-named files (deferred; need user confirmation)

These exist **only** in `nd/` and `nd-ml/`, with ambiguous names. Not included in
the first migration pass. Need Facundo's input before deciding if they should be
kept, renamed, or dropped:

| Filename | Size (bytes) | Observation |
|---|---|---|
| `L3.mat` | 228,274 | no bin size in name — likely an early/raw version |
| `L5.mat` | 213,335 | same |
| `L5_200msbin.mat` | 300,722 | slightly different size vs `L5_bins200ms_completo.mat` (302,264) — possibly an older preprocessing variant |
| `n31.mat` | 156,465 | lowercase naming; different size vs `N31_bins200ms_completo.mat` |
| `N59.mat` | 108,054 | no bin size in name |

**Question for Facundo:** do any of these represent data you still use, or are they
all superseded by the `*_bins200ms_completo.mat` versions? If unused, they move to
`_legacy/original-datasets/` instead of `data/raw/`.

## Migration plan

1. Create `data/raw/` inside the new structure.
2. Copy (not move) every file listed in **Authoritative catalog** above. Copying
   preserves the originals in the old directories for now — they can be deleted
   later once we confirm the new project works end-to-end.
3. Write `data/raw/README.md` that cross-references each file with region, animal,
   session, bin size, cohort, and experiments that use it.
4. Defer the **legacy-named files** decision until Facundo weighs in.

## Total authoritative dataset count

- Hippocampus: **8** datasets
- Expert piriform: **4** × 200ms + **4** × 100ms = **8** datasets
- Naive piriform: **2** processed + **2** original = **4** files (2 animals)

**Grand total for `data/raw/`: 20 files.**
