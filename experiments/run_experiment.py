#!/usr/bin/env python3
"""Run a hyperparameter search + nested-CV evaluation on one preprocessed dataset.

Loads an `(X, y, T)` pickle produced by `scripts/preprocess_all.py`, builds an
MLP model_space (single-output for `y.shape[1] == 1`, dual-output for `== 2`),
and calls `ibioml.tuner.run_study` with `inner_folds=1` (single hold-out per
outer fold, current thesis default).

Usage:
    # Full run (matches the lab's current all-datasets defaults):
    python experiments/run_experiment.py \\
        --pickle data/processed/piriform/Pir_L5_expert_bins200ms_withCtxt_flat.pickle \\
        --study-name mlp_pir_L5_dual

    # Local smoke (reduced params, same code path):
    python experiments/run_experiment.py \\
        --pickle data/processed/piriform/Pir_L5_expert_bins200ms_withCtxt_flat.pickle \\
        --study-name smoke_pir_L5_dual --smoke

Outputs go to `results/<study-name>/` by default.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from ibioml.models import MLPModel
from ibioml.tuner import run_study

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_NUM_TRIALS = 20
DEFAULT_NUM_EPOCHS = 200
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_FOLDS = 1
DEFAULT_BATCH_SIZE = 32
DEFAULT_ES_PATIENCE = 20

SMOKE_NUM_TRIALS = 3
SMOKE_NUM_EPOCHS = 50
SMOKE_OUTER_FOLDS = 2

# Optuna search space (copied verbatim from the lab's reference all-datasets script).
SEARCH_SPACE = {
    "hidden_size": (int, 128, 512, 64),      # (type, low, high, step)
    "num_layers":  (int, 1, 3),              # (type, low, high)
    "dropout":     (float, 0, 0.5),          # (type, low, high)
    "lr":          (float, 1e-8, 1e-3, "log"),  # (type, low, high, log-scale)
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pickle", required=True, type=Path,
        help="Path to the preprocessed (X, y, T) pickle.",
    )
    p.add_argument(
        "--study-name", required=True,
        help="Name of the Optuna study. Results go to <save-path>/<study-name>/.",
    )
    p.add_argument("--num-trials", type=int, default=DEFAULT_NUM_TRIALS)
    p.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    p.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    p.add_argument(
        "--inner-folds", type=int, default=DEFAULT_INNER_FOLDS,
        help="Inner-CV folds. Default 1 = single hold-out per outer fold.",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--es-patience", type=int, default=DEFAULT_ES_PATIENCE)
    p.add_argument(
        "--save-path", type=Path, default=ROOT / "results",
        help="Directory that will contain <study-name>/. Default: results/",
    )
    p.add_argument(
        "--device", choices=["cpu", "cuda", "auto"], default="auto",
        help="Torch device. 'auto' picks cuda if available, else cpu.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help=f"Reduce params to smoke-run levels "
             f"({SMOKE_NUM_TRIALS} trials, {SMOKE_NUM_EPOCHS} epochs, "
             f"{SMOKE_OUTER_FOLDS} outer folds). Overrides matching flags.",
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_pickle(path: Path):
    """Load an (X, y, T) tuple from disk with basic shape checks."""
    if not path.is_file():
        raise FileNotFoundError(f"Pickle not found: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, tuple) or len(obj) != 3:
        raise ValueError(
            f"Expected a 3-tuple (X, y, T) in {path}, got {type(obj).__name__} "
            f"of length {len(obj) if hasattr(obj, '__len__') else '?'}."
        )
    X, y, T = obj
    if X.shape[0] != y.shape[0] or X.shape[0] != T.shape[0]:
        raise ValueError(
            f"Row count mismatch: X={X.shape}, y={y.shape}, T={T.shape}"
        )
    if y.ndim != 2 or y.shape[1] not in (1, 2):
        raise ValueError(f"y must be (N, 1) or (N, 2); got {y.shape}")
    return X, y, T


def build_model_space(y_dim: int, device: torch.device, num_epochs: int,
                      es_patience: int, batch_size: int) -> dict:
    """Build the ibioml `model_space` dict for the given target dimensionality."""
    base = {
        "model_class": MLPModel,
        "output_size": y_dim,
        "device": device,
        "num_epochs": num_epochs,
        "es_patience": es_patience,
        "reg_type": None,
        "lambda_reg": None,
        "batch_size": batch_size,
        **SEARCH_SPACE,
    }
    if y_dim == 2:
        # Dual-output path uses one exclusive head per target
        # (matches the lab's reference all-datasets-experiment.py configuration).
        base["num_exclusive_layers"] = 1
    return base


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    """Mutate args in place to use the smoke-run presets."""
    args.num_trials = SMOKE_NUM_TRIALS
    args.num_epochs = SMOKE_NUM_EPOCHS
    args.outer_folds = SMOKE_OUTER_FOLDS


def main() -> int:
    args = parse_args()
    if args.smoke:
        apply_smoke_overrides(args)

    device = resolve_device(args.device)

    X, y, T = load_pickle(args.pickle)
    y_dim = y.shape[1]
    target_label = {1: "single", 2: "dual"}[y_dim]

    print("=" * 60)
    print(f"Study        : {args.study_name}")
    print(f"Pickle       : {args.pickle.relative_to(ROOT) if args.pickle.is_relative_to(ROOT) else args.pickle}")
    print(f"Shapes       : X={X.shape}, y={y.shape}, T={T.shape}")
    print(f"Target       : {target_label} (output_size={y_dim})")
    print(f"Device       : {device}")
    print(f"Outer folds  : {args.outer_folds}")
    print(f"Inner folds  : {args.inner_folds}")
    print(f"Optuna trials: {args.num_trials}")
    print(f"Epochs / patience: {args.num_epochs} / {args.es_patience}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Save path    : {args.save_path}/{args.study_name}/")
    if args.smoke:
        print("Smoke mode   : ON (reduced params)")
    print("=" * 60)

    model_space = build_model_space(
        y_dim=y_dim, device=device,
        num_epochs=args.num_epochs, es_patience=args.es_patience,
        batch_size=args.batch_size,
    )

    args.save_path.mkdir(parents=True, exist_ok=True)

    run_study(
        X=X, y=y, T=T,
        model_space=model_space,
        num_trials=args.num_trials,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        save_path=str(args.save_path),
        study_name=args.study_name,
    )

    print(f"\nDone. Results under: {args.save_path}/{args.study_name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
