#!/usr/bin/env python3
"""Build behaviour-only decoder pickles for the decoupling control (Batch 011).

The neural position decoder of the final pipeline (no tail 515° + Gaussian
smoothing σ=200 ms + window 0/1/0) sees the smoothed spike counts of the current
bin plus the context one-hot. This script writes, for the SAME kept bins, trial
markers and position target, a pickle whose features contain no neural data:

    X = [speed(t-2), speed(t-1), speed(t), speed(t+1), speed(t+2), ctx, not-ctx]

Speed neighbours are taken on the raw (unfiltered) recording, the same
convention the Gaussian smoothing uses, so the behavioural decoder gets at least
the temporal reach of the neural input (σ = 1 bin → weights ≈ 0.14 at ±2 bins,
≈ 0.01 at ±3). Training the same MLP / Optuna / folds on these pickles gives the
"speed + context reader" ceiling: what a decoder that only knew the animal's
running could reach. The ridge of `exploration/14-…` is linear and underestimates
that ceiling (the speed→position map is non-linear).

Folds are paired with the neural studies for free: `TrialKFold(5, shuffle,
seed=42)` depends only on the trial markers, which are copied verbatim.

With ``--include-neural`` the smoothed neural features of the source pickle are
prepended (``X = [neural…, speed(t-2…t+2), ctx, not-ctx]``): the "neural +
behaviour" decoder whose gain over the behaviour-only one measures the position
information the neurons carry beyond running (Batch 012).

Usage:
    uv run python scripts/build_behavioral_pickles.py \
        --src data/processed_notail_smooth_w010 --out data/processed_behav_speedctx_w010
    uv run python scripts/build_behavioral_pickles.py --include-neural \
        --out data/processed_neural_speedctx_w010
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ibioml.preprocess_data import (  # noqa: E402
    get_idx_by_low_performance,
    get_idx_by_out_of_corridor_trials,
    get_idx_by_position_tail,
    periods_immobility,
)

from analysis.loaders import load_mat  # noqa: E402

SPEED_LAGS = (-2, -1, 0, 1, 2)
POS_VARIANT = "_withCtxt_onlyPosition_flat.pickle"
# run_all_experiments discovers datasets by this marker file; only the pos target is run.
MARKER_VARIANT = "_withCtxt_flat.pickle"


def kept_raw_indices(mat: dict, cohort: str, tail_deg: float = 515.0) -> np.ndarray:
    """Raw-bin indices kept by the final pipeline (window 0/1/0 + tail), as in script 16."""
    pos = np.asarray(mat["position"], float).ravel()
    vel = np.asarray(mat["velocity"], float).ravel()
    tfb = np.asarray(mat["trialFinalBin"], int).ravel() - 1
    dprime = np.asarray(mat["dPrime"], float).ravel()
    thresh = 2 if cohort == "expert" else -1.1
    immob, trans = periods_immobility(vel, 1, 200, 4, tfb, 0, 0)
    removed = np.union1d(np.array(immob + trans, dtype=int),
                         get_idx_by_low_performance(dprime, tfb, thresh).astype(int))
    removed = np.union1d(removed, get_idx_by_out_of_corridor_trials(pos, tfb, 1500))
    removed = np.union1d(removed, np.where(np.isnan(pos))[0])
    removed = np.union1d(removed, get_idx_by_position_tail(pos, tfb, tail_deg, 0))
    return np.setdiff1d(np.arange(len(pos)), removed.astype(int))


def build_one(src_pkl: Path, mat_file: Path, out_dir: Path,
              include_neural: bool = False) -> tuple[int, int]:
    """Write the behavioural pickle (and discovery marker) for one dataset."""
    with src_pkl.open("rb") as f:
        X, y, trial_markers = pickle.load(f)  # noqa: S301 — our own artefact
    mat = load_mat(mat_file)
    cohort = "expert" if "_expert_" in src_pkl.name else "naive"
    kept = kept_raw_indices(mat, cohort)
    pos = np.asarray(mat["position"], float).ravel()
    if len(kept) != len(y) or not np.allclose(pos[kept], y[:, 0]):
        raise ValueError(f"{src_pkl.name}: reconstructed kept set does not match the pickle "
                         f"({len(kept)} vs {len(y)} rows)")
    vel = np.asarray(mat["velocity"], float).ravel()
    lag_idx = np.clip(kept[:, None] + np.array(SPEED_LAGS)[None, :], 0, len(vel) - 1)
    ctx = X[:, -2:]
    if not np.isin(ctx, (0.0, 1.0)).all():
        raise ValueError(f"{src_pkl.name}: last two columns are not the context one-hot")
    parts = [X[:, :-2]] if include_neural else []
    X_behav = np.concatenate([*parts, vel[lag_idx], ctx], axis=1)
    stem = src_pkl.name[: -len(POS_VARIANT)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in (POS_VARIANT, MARKER_VARIANT):
        with (out_dir / f"{stem}{variant}").open("wb") as f:
            pickle.dump((X_behav, y, trial_markers), f)
    return X_behav.shape


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, default=ROOT / "data" / "processed_notail_smooth_w010")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed_behav_speedctx_w010")
    p.add_argument("--include-neural", action="store_true",
                   help="Prepend the source pickle's (smoothed) neural features.")
    args = p.parse_args()
    for src_pkl in sorted(args.src.glob(f"*/*{POS_VARIANT}")):
        region_dir = src_pkl.parent.name
        mat_file = ROOT / "data" / "raw" / region_dir / (src_pkl.name[: -len(POS_VARIANT)] + ".mat")
        shape = build_one(src_pkl, mat_file, args.out / region_dir, args.include_neural)
        print(f"{src_pkl.name[: -len(POS_VARIANT)]:32s} X={shape}")


if __name__ == "__main__":
    main()
