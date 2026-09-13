# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Position/velocity decoupling — behavioural control decoder (Phase 0, part C)
#
# How much of the "position decoding" could come from behaviour alone? A ridge
# decoder predicts position from **no neural data at all**: the measured speed
# in the same 1 s window the MLP sees (5 bins, built on the raw recording
# before row filtering — same convention as `get_spikes_with_history`) plus
# the context one-hot. If the neural decoder does not clearly beat this
# behavioural ceiling, its R² can be explained by the running stereotypy; the
# margin above it is the part that requires neural information beyond speed.
#
# Three feature sets: context only (2), speed window (5), speed + context (7).
# CV: the same `TrialKFold(5, shuffle, seed=42)` partition as the neural
# studies — asserted identical — so folds are paired with the MLP results.
#
# Outputs under `results/plots/decoupling/behavioral-control/`:
# `summary_table.md/csv`, `r2_neural_vs_behavioral.svg`.

# %%
import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from ibioml.preprocess_data import (  # noqa: E402
    get_idx_by_low_performance,
    get_idx_by_out_of_corridor_trials,
    periods_immobility,
)
from ibioml.utils.splitters import TrialKFold  # noqa: E402

from analysis.loaders import CM_PER_DEGREE, load_mat  # noqa: E402
from analysis.results import (  # noqa: E402
    fold_predictions,
    fold_test_trials,
    load_final_results,
    mat_path,
    parse_study_name,
    pickle_path,
)

# %%
# --- config -----------------------------------------------------------------

DATASETS = os.environ.get(
    "DATASETS",
    "Pir_L3_expert,Pir_L5_expert,Pir_N31_expert,Pir_N59_expert,CA3_S18_expert,DG_S17_expert",
).split(",")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "processed"))
PLOT_DIR = Path(os.environ.get(
    "PLOT_DIR", ROOT / "results" / "plots" / "decoupling" / "behavioral-control"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 2            # speed bins before/after the current one (matches the MLP's 2/1/2)
ALPHAS = np.logspace(-2, 5, 15)
FEATURE_SETS = ["context only", "speed window", "speed + context"]
COLORS = {"neural MLP": "#004E89", "speed + context": "#E76F51",
          "speed window": "#E9C46A", "context only": "#7B848F"}
INK = "#333333"


# %%
# --- features from the raw recording (windows built before row filtering) -----

def kept_raw_indices(ctx, bins_before=2, bins_after=2):
    """Raw-bin indices surviving the baseline row filters (library logic; cf. script 12)."""
    n_raw = len(ctx["pos"])
    immob, trans = periods_immobility(ctx["vel"], 1, 200, 4, ctx["tfb"],
                                      bins_before, bins_after)
    removed = np.union1d(
        np.union1d(np.arange(bins_before), np.arange(n_raw - bins_after, n_raw)),
        np.union1d(np.array(immob + trans, dtype=int),
                   np.union1d(
                       get_idx_by_low_performance(ctx["dprime"], ctx["tfb"],
                                                  ctx["thresh"]).astype(int),
                       np.union1d(
                           get_idx_by_out_of_corridor_trials(ctx["pos"], ctx["tfb"], 1500),
                           np.where(np.isnan(ctx["pos"]))[0]))))
    return np.setdiff1d(np.arange(n_raw), removed.astype(int))


def load_dataset(stem_short):
    stem = f"{stem_short}_bins200ms"
    info = parse_study_name(f"mlp_{stem}_pos", "pos")
    mat = load_mat(mat_path(ROOT / "data" / "raw", info))
    ctx = {"pos": np.asarray(mat["position"], float).ravel(),
           "vel": np.asarray(mat["velocity"], float).ravel(),
           "tfb": np.asarray(mat["trialFinalBin"], int).ravel() - 1,
           "dprime": np.asarray(mat["dPrime"], float).ravel(),
           "thresh": 2 if info.cohort == "expert" else -1.1}
    rew = np.asarray(mat["rewCtxt"], float).ravel()
    with pickle_path(DATA_DIR, info).open("rb") as f:
        _, y, trial_markers = pickle.load(f)  # noqa: S301 — our own artefact
    kept = kept_raw_indices(ctx)
    if len(kept) != len(y) or not np.allclose(ctx["pos"][kept], y[:, 0]):
        raise ValueError(f"{stem}: kept set does not match the pickle")
    # speed window per kept bin, from the raw temporal neighbours
    offsets = np.arange(-WINDOW, WINDOW + 1)
    speed_win = ctx["vel"][kept[:, None] + offsets[None, :]]
    ctxt_onehot = np.stack([rew[kept], 1 - rew[kept]], axis=1)
    features = {"context only": ctxt_onehot,
                "speed window": speed_win,
                "speed + context": np.hstack([speed_win, ctxt_onehot])}
    return info, features, y[:, 0], trial_markers


# %%
# --- ridge with the neural studies' fold partition -----------------------------

def ridge_per_fold(features, y_pos, trial_markers, info):
    """Per-fold R²/MAE for each feature set, on the same folds as the MLP study."""
    neural_trials = fold_test_trials(RESULTS_DIR, info.study)
    splits = list(TrialKFold(n_splits=5, shuffle=True, random_state=42).split(trial_markers))
    rows = []
    for k, (train_idx, test_idx) in enumerate(splits, start=1):
        test_trials = sorted(int(t) for t in np.unique(trial_markers[test_idx]))
        assert test_trials == sorted(neural_trials[k - 1]), \
            f"{info.display} fold {k}: partition differs from the neural study"
        for fs in FEATURE_SETS:
            feats = features[fs]
            scaler = StandardScaler().fit(feats[train_idx])
            model = RidgeCV(alphas=ALPHAS).fit(scaler.transform(feats[train_idx]),
                                               y_pos[train_idx])
            pred = model.predict(scaler.transform(feats[test_idx]))
            rows.append({"Dataset": info.display, "model": fs, "fold": k,
                         "r2": r2_score(y_pos[test_idx], pred),
                         "mae": float(np.mean(np.abs(pred - y_pos[test_idx])) * CM_PER_DEGREE)})
    return rows


def neural_per_fold(info):
    final = load_final_results(RESULTS_DIR, info.study)
    rows = []
    for k, ((y_true, y_pred), r2) in enumerate(
            zip(fold_predictions(final, "position"), final["test_r2_scores"], strict=True),
            start=1):
        rows.append({"Dataset": info.display, "model": "neural MLP", "fold": k,
                     "r2": float(r2),
                     "mae": float(np.mean(np.abs(y_pred - y_true)) * CM_PER_DEGREE)})
    return rows


# %%
# --- run ------------------------------------------------------------------------

print("=" * 100)
print("DECOUPLING PHASE 0 — behavioural control decoder (ridge, no neural data)")
print("=" * 100)
all_rows = []
for stem_short in DATASETS:
    info, features, y_pos, trial_markers = load_dataset(stem_short)
    all_rows += neural_per_fold(info)
    all_rows += ridge_per_fold(features, y_pos, trial_markers, info)
    print(f"  ✅ {info.display}")

per_fold = pd.DataFrame(all_rows)
per_fold.to_csv(PLOT_DIR / "per_fold_metrics.csv", index=False, float_format="%.4f")
med = per_fold.groupby(["Dataset", "model"], sort=False)[["r2", "mae"]].median()


# %%
# --- table + figure -------------------------------------------------------------

def write_md(path):
    order = ["neural MLP", *FEATURE_SETS]
    lines = [
        "# Behavioural control decoder vs the neural position decoder",
        "",
        "Ridge on behaviour only (speed in the same 1 s window the MLP sees, plus the "
        "context one-hot), same TrialKFold partition as the neural studies (paired folds, "
        "asserted). Per-fold medians; MAE in cm; full test support. The gap between the "
        "neural MLP and 'speed + context' is the part of the decoding that requires neural "
        "information beyond the running stereotypy.",
        "",
        "| Dataset | " + " | ".join(f"R² {m}" for m in order) +
        " | MAE MLP | MAE speed+ctxt | margin ΔR² (MLP − speed+ctxt) |",
        "|---|" + "---:|" * (len(order) + 3),
    ]
    for ds in per_fold.Dataset.unique():
        r2s = [med.loc[(ds, m), "r2"] for m in order]
        sub = per_fold[per_fold.Dataset == ds].pivot(index="fold", columns="model", values="r2")
        margin = (sub["neural MLP"] - sub["speed + context"]).median()
        lines.append(
            f"| {ds} | " + " | ".join(f"{v:.3f}" for v in r2s) +
            f" | {med.loc[(ds, 'neural MLP'), 'mae']:.1f} "
            f"| {med.loc[(ds, 'speed + context'), 'mae']:.1f} | {margin:+.3f} |")
    Path(path).write_text("\n".join(lines) + "\n")


write_md(PLOT_DIR / "summary_table.md")
med.reset_index().to_csv(PLOT_DIR / "summary_table.csv", index=False, float_format="%.4f")
print(f"📁 {(PLOT_DIR / 'summary_table.md').relative_to(ROOT)}")

fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
names = list(per_fold.Dataset.unique())
x = np.arange(len(names))
order = ["neural MLP", "speed + context", "speed window", "context only"]
for j, m in enumerate(order):
    vals = [med.loc[(ds, m), "r2"] for ds in names]
    ax.bar(x + (j - 1.5) * 0.2, vals, width=0.19, color=COLORS[m], label=m)
ax.set_xticks(x, [n.replace("-Expert", "") for n in names], fontsize=9)
ax.axhline(0, color="0.25", linewidth=0.9)
ax.set_ylabel("R² (per-fold median, test)", fontsize=10, color=INK)
ax.set_title("Neural position decoder vs behaviour-only controls",
             fontsize=10.5, color=INK, loc="left")
ax.grid(axis="y", color="0.85", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=8.5, frameon=False)
fig.tight_layout()
out = PLOT_DIR / "r2_neural_vs_behavioral.svg"
fig.savefig(out, format="svg", bbox_inches="tight")
plt.close(fig)
print(f"📁 {out.relative_to(ROOT)}")

print()
print("=" * 100)
print("HEADLINE")
print("=" * 100)
for ds in per_fold.Dataset.unique():
    sub = per_fold[per_fold.Dataset == ds].pivot(index="fold", columns="model", values="r2")
    margin = (sub["neural MLP"] - sub["speed + context"]).median()
    wins = int((sub["neural MLP"] > sub["speed + context"]).sum())
    print(f"  {ds:<16} MLP {med.loc[(ds, 'neural MLP'), 'r2']:.3f} · "
          f"speed+ctxt {med.loc[(ds, 'speed + context'), 'r2']:.3f} · "
          f"margen ΔR² {margin:+.3f} ({wins}/5)")
