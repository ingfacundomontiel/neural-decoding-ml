"""Open topic T3 — CPU preview of the behaviour-only decoder (Batch 011 runs the MLP).

Same kept bins, targets and test folds as the final-pipeline neural studies;
features without neural data: speed(t−2…t+2) + context one-hot
(`data/processed_behav_speedctx_w010/`). Decoder: gradient-boosted trees, a
strong non-linear learner for 7 tabular features. Reports, per recording, the
test R² (fold median) of the neural MLP vs the behavioural decoder, and the
pre-context slope (pred vs true, 5–40 cm) of both — does running alone
reproduce the graded pre-context tracking of the expert piriform decoders?
"""
from __future__ import annotations

import contextlib
import io
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

import poster_data as pdata
from poster_style import FIG_DIR, REPO, setup

from analysis.loaders import CM_PER_DEGREE
from analysis.results import pickle_path

BEHAV = REPO / "data" / "processed_behav_speedctx_w010"
OUT = FIG_DIR / "t3"


def pre_slope(x: np.ndarray, p: np.ndarray) -> float:
    m = (x >= 5) & (x <= 40)
    return float(np.polyfit(x[m], p[m], 1)[0])


def behaviour_run(info, feats: slice) -> tuple[list[float], np.ndarray, np.ndarray]:
    with pickle_path(BEHAV, info, pdata.PICKLE_VARIANT["pos"]).open("rb") as f:
        X, y, trials = pickle.load(f)  # noqa: S301 — our own artefact
    X, y = X[:, feats], y[:, 0] * CM_PER_DEGREE
    r2s, xs, ps = [], [], []
    for test_trials in pdata.fold_trials(info.stem, "pos"):
        te = np.isin(trials, test_trials)
        mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                            early_stopping=True, random_state=0)
        mdl.fit(X[~te], y[~te])
        pred = mdl.predict(X[te])
        r2s.append(r2_score(y[te], pred))
        xs.append(y[te])
        ps.append(pred)
    return r2s, np.concatenate(xs), np.concatenate(ps)


def main() -> None:
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for info in pdata.studies("pos"):
        with contextlib.redirect_stdout(io.StringIO()):
            neural = pdata.load_run(info.stem, "pos")
        r2_sc, x_b, p_b = behaviour_run(info, slice(0, 7))
        r2_s, _, _ = behaviour_run(info, slice(0, 5))
        rows.append({"recording": info.display, "neural_r2": neural.median_r2,
                     "speed_ctx_r2": float(np.median(r2_sc)), "speed_r2": float(np.median(r2_s)),
                     "margin": neural.median_r2 - float(np.median(r2_sc)),
                     "folds_neural_wins": int(np.sum(np.array(neural.fold_r2) > np.array(r2_sc))),
                     "neural_pre_slope": pre_slope(neural.y_true, neural.y_pred),
                     "behav_pre_slope": pre_slope(x_b, p_b)})
        print(rows[-1])
    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "t3_behaviour_preview.csv", index=False)
    (OUT / "t3_behaviour_preview.txt").write_text(df.to_string(index=False))
    print(df.to_string(index=False))


# --- incremental test: does neural activity add to behaviour? (same learner) ----

def incremental(info) -> dict:
    """HGB on neural-only / behaviour-only / both, same folds; R² medians + per-zone MAE."""
    with pickle_path(BEHAV, info, pdata.PICKLE_VARIANT["pos"]).open("rb") as f:
        Xb, y, trials = pickle.load(f)  # noqa: S301
    with pickle_path(pdata.DATA, info, pdata.PICKLE_VARIANT["pos"]).open("rb") as f:
        Xn, y2, _ = pickle.load(f)  # noqa: S301
    assert np.array_equal(y, y2)
    y = y[:, 0] * CM_PER_DEGREE
    feats = {"neural": Xn[:, :-2], "behav": Xb, "both": np.hstack([Xn[:, :-2], Xb])}
    out = {k: [] for k in feats}
    preds = {k: np.empty_like(y) for k in feats}
    for test_trials in pdata.fold_trials(info.stem, "pos"):
        te = np.isin(trials, test_trials)
        for k, X in feats.items():
            mdl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                early_stopping=True, random_state=0)
            mdl.fit(X[~te], y[~te])
            preds[k][te] = mdl.predict(X[te])
            out[k].append(r2_score(y[te], preds[k][te]))
    row = {"recording": info.display}
    for k in feats:
        row[f"r2_{k}"] = float(np.median(out[k]))
    row["delta_both_minus_behav"] = row["r2_both"] - row["r2_behav"]
    row["folds_both_wins"] = int(np.sum(np.array(out["both"]) > np.array(out["behav"])))
    for zname, (lo, hi) in {"pre": (0, 44.7), "ctx": (44.7, 63.0)}.items():
        m = (y >= lo) & (y < hi)
        for k in ("behav", "both"):
            row[f"mae_{zname}_{k}"] = float(np.mean(np.abs(preds[k] - y)[m]))
    return row


def main_incremental() -> None:
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for info in pdata.studies("pos"):
        rows.append(incremental(info))
    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "t3_incremental.csv", index=False)
    (OUT / "t3_incremental.txt").write_text(df.to_string(index=False))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
    main_incremental()
