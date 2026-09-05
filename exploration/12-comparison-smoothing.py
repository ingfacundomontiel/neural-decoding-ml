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
# # Item 7 — Gaussian-smoothed spike counts vs the history window (2×2 on Pir_N31)
#
# Four runs on the same dataset (Batch 005 + the Batch 002 anchor):
#
# | run | activity | window |
# |---|---|---|
# | baseline | raw | 2/1/2 |
# | C | smoothed σ=200 ms | 2/1/2 |
# | B | raw | 0/1/0 |
# | A | smoothed σ=200 ms | 0/1/0 |
#
# The window size doubles as row-filter padding, so the 0/1/0 runs keep more
# bins (20,527) than the 2/1/2 ones (18,711) — and the 2/1/2 kept set is a
# strict subset of the 0/1/0 one. To put all four runs on **identical bins**,
# every metric here is computed on the **common support**: the raw-recording
# bins that survive the 2/1/2 filters. The mapping pooled-prediction → raw bin
# is reconstructed per run (`pooled_test_rows`, exact-verified) and the kept
# sets are recomputed from the raw `.mat` with the library's own filter
# functions (verified against the pickles). Folds are paired: the trial
# partition (seed 42) is identical across runs — asserted.
#
# Outputs under `results_smooth200_w010/plots/comparison/`: per-fold metrics,
# compact table, MAE-by-position overlay of the four runs, stdout headline.

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

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from ibioml.preprocess_data import (  # noqa: E402
    get_idx_by_low_performance,
    get_idx_by_out_of_corridor_trials,
    periods_immobility,
)

from analysis.loaders import CM_PER_DEGREE, CONTEXT_END_CM, CONTEXT_START_CM, load_mat  # noqa: E402
from analysis.metrics import binned_error, error_summary  # noqa: E402
from analysis.results import (  # noqa: E402
    fold_predictions,
    fold_test_trials,
    load_final_results,
    parse_study_name,
    pool_folds,
    pooled_test_rows,
)

# %%
# --- config -----------------------------------------------------------------

STEM = os.environ.get("STEM", "Pir_N31_expert_bins200ms")
STUDY = f"mlp_{STEM}_pos"
INFO = parse_study_name(STUDY, "pos")
RAW_MAT = ROOT / "data" / "raw" / "piriform" / f"{STEM}.mat"

RUNS = {
    "baseline raw·2/1/2": {"results": "results", "data": "data/processed",
                           "window": (2, 2), "color": "#7B848F"},
    "C smooth·2/1/2": {"results": "results_smooth200_w212",
                       "data": "data/processed_smooth200_w212",
                       "window": (2, 2), "color": "#2A9D8F"},
    "B raw·0/1/0": {"results": "results_raw_w010", "data": "data/processed_raw_w010",
                    "window": (0, 0), "color": "#E76F51"},
    "A smooth·0/1/0": {"results": "results_smooth200_w010",
                       "data": "data/processed_smooth200_w010",
                       "window": (0, 0), "color": "#004E89"},
}
PLOT_DIR = Path(os.environ.get("PLOT_DIR",
                               ROOT / "results_smooth200_w010" / "plots" / "comparison"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)
BIN_CM, MIN_COUNT = 2.0, 30
INK = "#333333"
GRID = dict(color="0.85", linewidth=0.6)


# %%
# --- reconstruct each config's kept raw bins (verified against the pickles) --

mat = load_mat(RAW_MAT)
pos_raw = np.asarray(mat["position"], dtype=float).ravel()
vel_raw = np.asarray(mat["velocity"], dtype=float).ravel()
tfb = np.asarray(mat["trialFinalBin"], dtype=int).ravel() - 1
dprime = np.asarray(mat["dPrime"], dtype=float).ravel()
n_raw = len(pos_raw)


def kept_raw_indices(bins_before, bins_after):
    """Raw-bin indices surviving the row filters for this window size (library logic)."""
    immob, trans = periods_immobility(vel_raw, 1, 200, 4, tfb, bins_before, bins_after)
    removed = np.union1d(
        np.union1d(np.arange(bins_before), np.arange(n_raw - bins_after, n_raw)),
        np.union1d(np.array(immob + trans, dtype=int),
                   np.union1d(get_idx_by_low_performance(dprime, tfb, 2).astype(int),
                              np.union1d(get_idx_by_out_of_corridor_trials(pos_raw, tfb, 1500),
                                         np.where(np.isnan(pos_raw))[0]))))
    return np.setdiff1d(np.arange(n_raw), removed.astype(int))


def load_run(name, cfg):
    pkl = ROOT / cfg["data"] / "piriform" / f"{STEM}_withCtxt_onlyPosition_flat.pickle"
    with pkl.open("rb") as f:
        _, y, trial_markers = pickle.load(f)  # noqa: S301 — our own preprocess artefact
    kept = kept_raw_indices(*cfg["window"])
    if len(kept) != len(y) or not np.allclose(pos_raw[kept], y[:, 0]):
        raise ValueError(f"{name}: reconstructed kept set does not match the pickle")
    rows = pooled_test_rows(ROOT / cfg["results"], STUDY, y[:, 0], trial_markers)
    final = load_final_results(ROOT / cfg["results"], STUDY)
    folds = [(t * CM_PER_DEGREE, p * CM_PER_DEGREE)
             for t, p in fold_predictions(final, "position")]
    y_true, y_pred, fold_id = pool_folds(folds)
    return {"y_true": y_true, "y_pred": y_pred, "fold_id": fold_id,
            "raw_idx": kept[rows],
            "trials": fold_test_trials(ROOT / cfg["results"], STUDY),
            "r2_folds": final["test_r2_scores"], **cfg}


print("=" * 100)
print(f"ITEM 7 — 2×2 smoothing × window on {INFO.display}")
print("=" * 100)
runs = {name: load_run(name, cfg) for name, cfg in RUNS.items()}
ref_trials = runs["baseline raw·2/1/2"]["trials"]
for name, d in runs.items():
    assert d["trials"] == ref_trials, f"{name}: fold partition differs from baseline"
print("fold partitions identical across the four runs ✓")

kept22 = kept_raw_indices(2, 2)
for name, d in runs.items():
    d["common"] = np.isin(d["raw_idx"], kept22)
    print(f"  {name:<20} pooled bins {len(d['y_true']):>6,} · on common support "
          f"{int(d['common'].sum()):>6,}")


# %%
# --- per-fold metrics on the common support ----------------------------------

rows = []
for name, d in runs.items():
    for k in np.unique(d["fold_id"]):
        m = (d["fold_id"] == k) & d["common"]
        rows.append({"run": name, "fold": int(k),
                     **error_summary(d["y_true"], d["y_pred"], m)})
per_fold = pd.DataFrame(rows)
per_fold.to_csv(PLOT_DIR / "per_fold_metrics.csv", index=False, float_format="%.4f")

med = per_fold.groupby("run", sort=False)[["n", "r2", "rmse", "mae"]].median()


def paired(run_a, run_b, metric):
    a = per_fold[per_fold.run == run_a].set_index("fold")[metric]
    b = per_fold[per_fold.run == run_b].set_index("fold")[metric]
    delta = a - b
    better = (delta > 0) if metric == "r2" else (delta < 0)
    return float(delta.median()), int(better.sum()), len(delta)


def fmt_pair(run_a, run_b, metric, nd):
    d, w, n = paired(run_a, run_b, metric)
    return f"{d:+.{nd}f} ({w}/{n})"


CONTRASTS = [
    ("C smooth·2/1/2", "baseline raw·2/1/2", "smoothing, window kept"),
    ("A smooth·0/1/0", "B raw·0/1/0", "smoothing, no window"),
    ("B raw·0/1/0", "baseline raw·2/1/2", "dropping the window (raw)"),
    ("A smooth·0/1/0", "C smooth·2/1/2", "dropping the window (smoothed)"),
    ("A smooth·0/1/0", "baseline raw·2/1/2", "the proposal vs the anchor"),
]


def write_md(path):
    lines = [
        f"# Item 7 — smoothing × window on {INFO.display} (common support)",
        "",
        "All four runs evaluated on the same bins (the 2/1/2 kept set; the 0/1/0 runs' "
        "extra padding bins are excluded). Per-fold medians, 5 paired folds. MSE = RMSE².",
        "",
        "| run | n bins | R² | RMSE | MAE | MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in RUNS:
        r = med.loc[name]
        lines.append(f"| {name} | {int(r['n']):,} | {r['r2']:.3f} | {r['rmse']:.1f} "
                     f"| {r['mae']:.1f} | {r['rmse'] ** 2:.0f} |")
    lines += ["", "## Paired contrasts (Δ = first − second; folds won by the first)", "",
              "| contrast | Δ R² | Δ RMSE | Δ MAE |", "|---|---:|---:|---:|"]
    for a, b, label in CONTRASTS:
        lines.append(f"| {label}: {a} − {b} | {fmt_pair(a, b, 'r2', 3)} "
                     f"| {fmt_pair(a, b, 'rmse', 1)} | {fmt_pair(a, b, 'mae', 1)} |")
    Path(path).write_text("\n".join(lines) + "\n")


write_md(PLOT_DIR / "comparison_table.md")
med.to_csv(PLOT_DIR / "comparison_table.csv", float_format="%.4f")
print(f"📁 {PLOT_DIR / 'comparison_table.md'}")


# %%
# --- MAE-by-position overlay (common support) --------------------------------

def overlay(save_path):
    hi = float(np.ceil(np.percentile(runs["baseline raw·2/1/2"]["y_true"], 99.9) / 10) * 10)
    edges = np.arange(0, hi + BIN_CM, BIN_CM)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    for name, d in runs.items():
        tbl = binned_error(d["y_true"][d["common"]], d["y_pred"][d["common"]],
                           edges, min_count=MIN_COUNT)
        ax.plot(tbl["center"], tbl["mae"], color=d["color"], linewidth=1.7, label=name)
    ax.axvspan(CONTEXT_START_CM, CONTEXT_END_CM, color="gold", alpha=0.15, lw=0)
    ax.axvline(CONTEXT_END_CM, color="0.35", linewidth=0.8, linestyle=":")
    ax.set_xlabel("True position (cm)", fontsize=10, color=INK)
    ax.set_ylabel("MAE (cm)", fontsize=10, color=INK)
    ax.set_title(f"{INFO.display} — MAE by position, four runs on the common support · "
                 "test predictions, 5 folds pooled", fontsize=10.5, color=INK, loc="left")
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"📁 {save_path}")


overlay(PLOT_DIR / "mae_by_position_overlay.svg")


# %%
# --- headline -----------------------------------------------------------------

print()
print("=" * 100)
print("HEADLINE — per-fold medians on the common support")
print("=" * 100)
for name in RUNS:
    r = med.loc[name]
    print(f"  {name:<20} R² {r['r2']:.3f} · RMSE {r['rmse']:5.1f} · MAE {r['mae']:.1f}")
print()
for a, b, label in CONTRASTS:
    print(f"  {label:<32} ΔR² {fmt_pair(a, b, 'r2', 3):>14} · "
          f"ΔRMSE {fmt_pair(a, b, 'rmse', 1):>12} · ΔMAE {fmt_pair(a, b, 'mae', 1):>12}")
