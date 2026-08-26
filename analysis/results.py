"""Read decoding-study outputs written by ``ibioml.tuner.run_study``.

Layout (see ``scripts/run_all_experiments.py``)::

    <results_dir>/mlp_<REGION>_<ID>_<cohort>_bins200ms_<target>/
        final_results.json          # per-fold R² + de-standardised predictions
        training_results/fold_<k>/  # per-fold Optuna artefacts

``target`` is one of ``pos`` / ``vel`` (single-output studies) or ``dual``.
``final_results.json`` stores ``predictions_per_fold`` and
``true_values_per_fold`` already inverse-transformed to the pickle's units:
raw encoder **degrees** for position (convert with
:data:`analysis.loaders.CM_PER_DEGREE`) and cm/s for velocity. Single-output
studies store flat lists per fold; dual studies store
``{"position": [...], "velocity": [...]}`` per fold.

The outer folds are disjoint trial partitions (``TrialKFold``), so pooling
the five folds yields exactly one out-of-sample prediction per surviving bin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REGION_DISPLAY = {"Pir": "Pir", "CA3": "CA3", "DG": "DG"}
COHORT_DISPLAY = {"expert": "Expert", "naive": "Naive"}
COHORT_TYPE_SUFFIX = {"expert": "_experto", "naive": "_naive"}
TARGETS = ("pos", "vel", "dual")
BIN_TAG = "_bins200ms"


@dataclass(frozen=True)
class StudyInfo:
    """Identity of one study directory, parsed from its name."""

    study: str   # directory name, e.g. "mlp_CA3_S18_expert_bins200ms_pos"
    stem: str    # dataset stem, e.g. "CA3_S18_expert_bins200ms"
    region: str  # Pir | CA3 | DG
    animal: str  # e.g. S18
    cohort: str  # expert | naive
    target: str  # pos | vel | dual

    @property
    def display(self) -> str:
        """Human label used across the plots, e.g. ``CA3-Expert-S18``."""
        return f"{REGION_DISPLAY[self.region]}-{COHORT_DISPLAY[self.cohort]}-{self.animal}"

    @property
    def dtype(self) -> str:
        """Dataset-type key used by the shared colour scheme, e.g. ``CA3_experto``."""
        prefix = "pir" if self.region == "Pir" else self.region
        return f"{prefix}{COHORT_TYPE_SUFFIX[self.cohort]}"


def parse_study_name(name: str, target: str) -> StudyInfo | None:
    """Parse ``mlp_<REGION>_<ID>_<cohort>_bins200ms_<target>``; None if it doesn't match."""
    prefix, suffix = "mlp_", f"_{target}"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    core = name[len(prefix):-len(suffix)]
    if not core.endswith(BIN_TAG):
        return None
    stem = core
    parts = core[: -len(BIN_TAG)].split("_")
    if len(parts) < 3:
        return None
    region, cohort, animal = parts[0], parts[-1], "_".join(parts[1:-1])
    if region not in REGION_DISPLAY or cohort not in COHORT_DISPLAY:
        return None
    return StudyInfo(study=name, stem=stem, region=region, animal=animal,
                     cohort=cohort, target=target)


def discover_studies(results_dir: str | Path, target: str) -> list[StudyInfo]:
    """Every ``mlp_*_<target>`` study under *results_dir* that has a ``final_results.json``."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
    root = Path(results_dir)
    if not root.is_dir():
        return []
    found = []
    for entry in sorted(root.iterdir()):
        if not (entry / "final_results.json").is_file():
            continue
        info = parse_study_name(entry.name, target)
        if info is not None:
            found.append(info)
    return found


def load_final_results(results_dir: str | Path, study: str) -> dict:
    """Load ``final_results.json`` for one study."""
    path = Path(results_dir) / study / "final_results.json"
    with path.open() as f:
        return json.load(f)


def fold_r2_scores(final: dict, variable: str = "position") -> list[float]:
    """Per-outer-fold test R² for *variable*, whichever study type produced *final*."""
    if "test_r2_scores_pos" in final:  # dual study
        key = "test_r2_scores_pos" if variable == "position" else "test_r2_scores_vel"
        return list(final[key])
    return list(final["test_r2_scores"])


def fold_predictions(final: dict,
                     variable: str = "position") -> list[tuple[np.ndarray, np.ndarray]]:
    """``[(y_true, y_pred), ...]`` per outer fold, in the pickle's original units.

    Works for both single-output (flat lists) and dual (dict per fold) studies.
    """
    out = []
    for true_f, pred_f in zip(final["true_values_per_fold"], final["predictions_per_fold"],
                              strict=True):
        if isinstance(true_f, dict):
            true_f, pred_f = true_f[variable], pred_f[variable]
        y_true = np.asarray(true_f, dtype=float).ravel()
        y_pred = np.asarray(pred_f, dtype=float).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError(f"fold shape mismatch: true={y_true.shape} pred={y_pred.shape}")
        out.append((y_true, y_pred))
    return out


def pool_folds(
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate folds → ``(y_true, y_pred, fold_id)`` with 1-based fold ids."""
    y_true = np.concatenate([t for t, _ in folds])
    y_pred = np.concatenate([p for _, p in folds])
    fold_id = np.concatenate([np.full(len(t), k + 1) for k, (t, _) in enumerate(folds)])
    return y_true, y_pred, fold_id
