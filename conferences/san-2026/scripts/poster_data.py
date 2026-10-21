"""Data access for the SAN 2026 poster figures — final pipeline only.

Final pipeline (plan.md, "Results pipeline"): position-tail filter 515° +
Gaussian smoothing σ=200 ms + window 0/1/0. Position = Batch 007, velocity =
Batch 008, both under ``results_notail_smooth_w010/`` with identical kept bins
and trial folds.
"""
from __future__ import annotations

import pickle
import statistics
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from poster_style import REPO, dataset_sort_key

from analysis.loaders import CM_PER_DEGREE, load_mat
from analysis.results import (
    StudyInfo,
    discover_studies,
    fold_predictions,
    fold_test_trials,
    load_final_results,
    mat_path,
    pickle_path,
    pool_folds,
    pooled_test_rows,
)
from ibioml.preprocess_data import (
    get_idx_by_low_performance,
    get_idx_by_out_of_corridor_trials,
    get_idx_by_position_tail,
    periods_immobility,
)

RESULTS = REPO / "results_notail_smooth_w010"
DATA = REPO / "data" / "processed_notail_smooth_w010"
RAW = REPO / "data" / "raw"
PICKLE_VARIANT = {"pos": "_withCtxt_onlyPosition_flat.pickle",
                  "vel": "_withCtxt_onlyVelocity_flat.pickle"}
TAIL_DEG = 515.0
# The poster shows the 14 recordings already presented at SfN 2025 (decided 2026-09-27).
EXCLUDED_STEMS = {"Pir_N59_expert_bins200ms", "DG_S24_naive_bins200ms", "DG_S41_naive_bins200ms"}


@dataclass
class Run:
    info: StudyInfo
    fold_r2: list[float]
    y_true: np.ndarray      # pooled test targets, cm (pos) or cm/s (vel)
    y_pred: np.ndarray
    fold_id: np.ndarray
    raw_idx: np.ndarray     # raw-recording bin of each pooled test row
    trial: np.ndarray       # trial id of each pooled test row
    n_features: int

    @property
    def median_r2(self) -> float:
        return statistics.median(self.fold_r2)


def studies(target: str = "pos") -> list[StudyInfo]:
    """The poster's final-pipeline studies, sorted region → cohort (naive first) → animal."""
    found = [i for i in discover_studies(RESULTS, target) if i.stem not in EXCLUDED_STEMS]
    return sorted(found, key=lambda i: dataset_sort_key(i.region, i.cohort, i.animal))


@lru_cache(maxsize=None)
def raw_recording(stem: str, region: str, cohort: str) -> dict:
    info = StudyInfo(study="", stem=stem, region=region, animal="", cohort=cohort, target="pos")
    mat = load_mat(mat_path(RAW, info))
    rec = {k: np.asarray(mat[k], float).ravel() for k in
           ("position", "velocity", "trialFinalBin", "dPrime", "rewCtxt")}
    rec["spikes"] = np.asarray(mat["neuronActivity"], float)   # bins × neurons (raw counts)
    return rec


def kept_raw_indices(rec: dict, cohort: str) -> np.ndarray:
    """Raw bins kept by the final pipeline (same logic as exploration/16, window 0/1/0)."""
    pos, vel = rec["position"], rec["velocity"]
    tfb = rec["trialFinalBin"].astype(int) - 1
    thresh = 2 if cohort == "expert" else -1.1
    immob, trans = periods_immobility(vel, 1, 200, 4, tfb, 0, 0)
    removed = np.union1d(np.array(immob + trans, dtype=int),
                         get_idx_by_low_performance(rec["dPrime"], tfb, thresh).astype(int))
    removed = np.union1d(removed, get_idx_by_out_of_corridor_trials(pos, tfb, 1500))
    removed = np.union1d(removed, np.where(np.isnan(pos))[0])
    removed = np.union1d(removed, get_idx_by_position_tail(pos, tfb, TAIL_DEG, 0))
    return np.setdiff1d(np.arange(len(pos)), removed.astype(int))


@lru_cache(maxsize=None)
def load_run(stem: str, target: str = "pos") -> Run:
    info = next(i for i in studies(target) if i.stem == stem)
    with pickle_path(DATA, info, PICKLE_VARIANT[target]).open("rb") as f:
        X, y, trial_markers = pickle.load(f)  # noqa: S301 — our own artefact
    rec = raw_recording(info.stem, info.region, info.cohort)
    kept = kept_raw_indices(rec, info.cohort)
    raw_target = rec["position" if target == "pos" else "velocity"]
    if len(kept) != len(y) or not np.allclose(raw_target[kept], y[:, 0]):
        raise ValueError(f"{info.display}/{target}: reconstructed kept set != pickle")
    final = load_final_results(RESULTS, info.study)
    rows = pooled_test_rows(RESULTS, info.study, y[:, 0], trial_markers)
    scale = CM_PER_DEGREE if target == "pos" else 1.0
    y_true, y_pred, fold_id = pool_folds(
        [(t * scale, p * scale) for t, p in fold_predictions(final)])
    return Run(info=info, fold_r2=list(final["test_r2_scores"]), y_true=y_true,
               y_pred=y_pred, fold_id=fold_id, raw_idx=kept[rows],
               trial=np.asarray(trial_markers)[rows], n_features=X.shape[1])


def fold_trials(stem: str, target: str = "pos") -> list[list[int]]:
    info = next(i for i in studies(target) if i.stem == stem)
    return fold_test_trials(RESULTS, info.study)
