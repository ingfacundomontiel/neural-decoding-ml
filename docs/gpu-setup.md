# GPU host setup (lab GPU host, via SSH)

Bootstrapping the thesis repo on a remote host so the preprocessing + training
pipeline runs end-to-end. Assumes the host already has:

- Python 3.10+ (`python3 --version`).
- `git`.
- `rsync` (present on almost every Unix).
- CUDA toolkit + GPU drivers (for GPU training). CPU-only works too, just slower.

The pipeline never re-downloads the raw `.mat` files — you copy them from your
laptop over `rsync`. Everything else is in git.

---

## 1. Clone the thesis with the submodule

```bash
git clone --recurse-submodules https://github.com/<user>/neural-decoding-ml.git thesis
cd thesis
```

If you cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

Verify the `ibioml` submodule landed at the pinned commit:

```bash
git -C external/ibioml log HEAD --oneline -1
# Should show acbbf3e or later (see specs/decisions.md D-015).
```

## 2. Create a virtualenv and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -e external/ibioml
pip install -r requirements.txt   # if the file exists — otherwise install from pyproject
```

Sanity check:

```bash
python -c "import ibioml; print(ibioml.__version__)"
# Expected: 0.1.6 (or newer, depending on the current pin)
```

## 3. Copy the raw `.mat` files from your laptop

The `.mat` files under `data/raw/<region>/*.mat` are **not tracked in git** (see
`specs/decisions.md` D-013 — lab recordings are not ours to redistribute). You
have to copy them by hand from your machine into the GPU host. From your
laptop:

```bash
rsync -avP data/raw/ <user>@<gpu-host>:~/thesis/data/raw/
```

`data/raw/README.md` documents the expected directory layout (one subdirectory
per brain region). The preprocess script iterates over
`data/raw/<region>/*_bins200ms.mat`, so anything missing there simply gets
skipped.

## 4. Run the preprocess batch

Preprocess every dataset with the current config (`bins_before=2`,
`bins_after=2`, `firingMinimo=0.1 Hz`, `threshDPrime` derived from filename):

```bash
python scripts/preprocess_all.py
```

Outputs (all under `data/processed/`, not tracked in git):

- `<region>/<dataset>_*.pickle` — 12 pickle variants per dataset.
- `preprocess_report.csv` — machine-readable summary.
- `preprocess_report.md` — human-readable summary with per-dataset table.
- `README.md` — records the config, thesis commit, and `ibioml` commit used.
- `logs/<dataset>.log` — captured stdout per dataset.

The batch is **idempotent**: it skips any dataset whose canonical output pickle
already exists. Pass `--force` to reprocess anyway, or `--only <substring>` to
target a single dataset (useful for a quick sanity check after copying a new
`.mat`).

### Filter variants (Phase 5 Item 6 and later)

A preprocessing variant must never overwrite the baseline pickles. Give it its
own tree with `--out-dir`, and point the batch runner at it with
`--data-dir` / `--results-dir` so the results land in a sibling of `results/`:

```bash
python scripts/preprocess_all.py --th-position-end 515 --out-dir data/processed_posend515
python scripts/run_all_experiments.py --targets pos --only Pir_N31,CA3_S18,DG_S17 \
    --data-dir data/processed_posend515 --results-dir results_posend515
```

Both trees are gitignored (`data/processed_*/`, `results_*/`). Bring the
results back with `rsync` the same way as `results/`, and read them with
`RESULTS_DIR=results_posend515 python exploration/08-position-error-analysis.py`.

## 5. Run one training experiment

Full config, dual output (position + velocity), 5 outer folds, Optuna with
20 trials:

```bash
python experiments/run_experiment.py \
    --pickle data/processed/piriform/Pir_L5_expert_bins200ms_withCtxt_flat.pickle \
    --study-name mlp_pir_L5_dual
```

The runner picks the target dimensionality (single vs dual) from the pickle's
`y.shape[1]`, and auto-detects the device (`cuda` if available, else `cpu`).
Results go to `results/<study-name>/`.

To iterate faster while wiring up a new dataset, add `--smoke` — reduces the
run to 3 Optuna trials, 50 epochs, 2 outer folds. Same code path, so if the
smoke passes, the full run will at least start correctly:

```bash
python experiments/run_experiment.py \
    --pickle <pickle> --study-name smoke_<x> --smoke
```

Override individual params too:

```bash
python experiments/run_experiment.py \
    --pickle <pickle> --study-name <name> \
    --num-trials 30 --num-epochs 300 --outer-folds 5 \
    --batch-size 64
```

## 6. Bumping the `ibioml` submodule when the library advances

When the lab maintainers merge something new to `mariburginlab-labPrograms/IBioML` main:

```bash
cd external/ibioml
git fetch upstream
git checkout main
git pull upstream main
cd -

git add external/ibioml
git commit -m "chore: bump ibioml submodule to <short-sha>"
git push
```

Then on any other host (your laptop, another GPU box), a `git pull` +
`git submodule update --init --recursive` picks up the new pin.

Rule of thumb: **before running a fresh batch, pull the thesis + update the
submodule** so the two hosts are in sync.

## Troubleshooting

- **`ImportError: No module named 'ibioml'`**: you forgot to activate the venv,
  or `pip install -e external/ibioml` didn't run. Re-activate and re-install.
- **`FileNotFoundError` at preprocess time**: the corresponding `.mat` is
  missing. Re-run the `rsync` step, or `ls data/raw/<region>/` to confirm.
- **CUDA out of memory**: reduce `--batch-size` (default 32; try 16 or 8) or
  reduce `--num-epochs` for the sweep.
- **Pipeline stalls / no logs**: `preprocess_data` is chatty per dataset; check
  `data/processed/logs/<dataset>.log` for the last thing it printed.
