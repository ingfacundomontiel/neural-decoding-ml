"""Decoding-error metrics in physical units.

Every function takes ``y_true`` / ``y_pred`` already converted to the unit
you want to report (cm for position, cm/s for velocity). Rationale for
reporting RMSE / MAE in cm as the primary metric and R² as secondary is in
``specs/decisions.md`` (D-018): R² is the MSE normalised by the variance of
the *evaluation set*, so it moves whenever the evaluation support changes
even if the errors do not. Compare models on the **same bins**.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(np.abs(y_pred - y_true)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error (pred − true). Positive = over-prediction."""
    return float(np.mean(y_pred - y_true))


def chance_rmse(y_true: np.ndarray) -> float:
    """RMSE of always predicting the mean of *y_true* (= its population std).

    Same information as R² (``R² = 1 − (RMSE / chance_rmse)²`` on the same
    bins) but in physical units, so the gain over chance is readable.
    """
    return float(np.std(y_true))


def error_summary(y_true: np.ndarray, y_pred: np.ndarray,
                  mask: np.ndarray | None = None) -> dict:
    """All scalar metrics on the (optionally masked) bins, plus the bin count."""
    if mask is not None:
        y_true, y_pred = y_true[mask], y_pred[mask]
    n = int(len(y_true))
    if n < 2:
        return {"n": n, "r2": np.nan, "rmse": np.nan, "mae": np.nan,
                "median_ae": np.nan, "bias": np.nan, "chance_rmse": np.nan}
    return {
        "n": n,
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "median_ae": median_ae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "chance_rmse": chance_rmse(y_true),
    }


def binned_error(y_true: np.ndarray, y_pred: np.ndarray, edges: np.ndarray,
                 min_count: int = 30) -> pd.DataFrame:
    """Error as a function of the *true* value, binned by *edges*.

    One row per bin: centre, count, median / q25 / q75 of the signed error,
    MAE, RMSE. Bins with fewer than *min_count* samples get NaN statistics
    (count is still reported) so sparse regions don't produce noisy curves.
    """
    err = y_pred - y_true
    idx = np.digitize(y_true, edges) - 1
    rows = []
    for i in range(len(edges) - 1):
        sel = err[idx == i]
        row = {"lo": edges[i], "hi": edges[i + 1],
               "center": 0.5 * (edges[i] + edges[i + 1]), "n": int(sel.size)}
        if sel.size >= min_count:
            q25, q50, q75 = np.percentile(sel, [25, 50, 75])
            row.update(median_signed=q50, q25=q25, q75=q75,
                       mae=float(np.mean(np.abs(sel))),
                       rmse=float(np.sqrt(np.mean(sel ** 2))))
        else:
            row.update(median_signed=np.nan, q25=np.nan, q75=np.nan,
                       mae=np.nan, rmse=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def zone_summary(y_true: np.ndarray, y_pred: np.ndarray,
                 zones: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """:func:`error_summary` per named zone ``{name: (lo, hi)}`` on the true value (lo ≤ y < hi).

    Adds ``pct_bins`` (share of samples in the zone) and ``pct_sq_error`` (share of
    the total squared error the zone contributes). A zone whose ``pct_sq_error``
    exceeds its ``pct_bins`` concentrates error.
    """
    sq = (y_pred - y_true) ** 2
    total_sq = float(np.sum(sq)) or np.nan
    rows = []
    for name, (lo, hi) in zones.items():
        mask = (y_true >= lo) & (y_true < hi)
        rows.append({"zone": name, "lo": lo, "hi": hi,
                     "pct_bins": 100 * mask.mean(),
                     "pct_sq_error": 100 * float(np.sum(sq[mask])) / total_sq,
                     **error_summary(y_true, y_pred, mask)})
    return pd.DataFrame(rows)


def per_fold_summary(folds: list[tuple[np.ndarray, np.ndarray]],
                     mask_fn=None) -> pd.DataFrame:
    """:func:`error_summary` per fold; *mask_fn(y_true)* optionally restricts the bins."""
    rows = []
    for k, (y_true, y_pred) in enumerate(folds, start=1):
        mask = mask_fn(y_true) if mask_fn is not None else None
        rows.append({"fold": k, **error_summary(y_true, y_pred, mask)})
    return pd.DataFrame(rows)
