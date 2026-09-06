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
# # Item 7 — Gaussian-smoothed spike counts vs the history window (2×2)
#
# Four runs per dataset (Batches 005/006 + the Batch 002 anchor):
#
# | run | activity | window |
# |---|---|---|
# | baseline | raw | 2/1/2 |
# | C | smoothed σ=200 ms | 2/1/2 |
# | B | raw | 0/1/0 |
# | A | smoothed σ=200 ms | 0/1/0 |
#
# The window size doubles as row-filter padding, so the 0/1/0 runs keep more
# bins than the 2/1/2 ones — and the 2/1/2 kept set is a strict subset of the
# 0/1/0 one. To put the four runs on **identical bins**, every metric is
# computed on the **common support**: the raw-recording bins that survive the
# 2/1/2 filters. The mapping pooled-prediction → raw bin is reconstructed per
# run (`pooled_test_rows`, exact-verified) and the kept sets are recomputed
# from the raw `.mat` with the library's own filter functions (verified
# against the pickles; `threshDPrime` follows the cohort: 2 expert / −1.1
# naive). Folds are paired — the trial partition is identical across runs
# (asserted).
#
# Datasets: every study present in all four results trees (auto-discovered).
# Outputs under `results_smooth200_w010/plots/comparison/`: per-fold metrics,
# per-dataset + aggregate tables, one MAE-by-position overlay per dataset.

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
    discover_studies,
    fold_predictions,
    fold_test_trials,
    load_final_results,
    mat_path,
    pickle_path,
    pool_folds,
    pooled_test_rows,
)

# %%
# --- config -----------------------------------------------------------------

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
CONTRASTS = [
    ("C smooth·2/1/2", "baseline raw·2/1/2", "smoothing, window kept"),
    ("A smooth·0/1/0", "B raw·0/1/0", "smoothing, no window"),
    ("B raw·0/1/0", "baseline raw·2/1/2", "dropping the window (raw)"),
    ("A smooth·0/1/0", "C smooth·2/1/2", "dropping the window (smoothed)"),
    ("A smooth·0/1/0", "baseline raw·2/1/2", "the proposal vs the anchor"),
]
PLOT_DIR = Path(os.environ.get("PLOT_DIR",
                               ROOT / "results_smooth200_w010" / "plots" / "comparison"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)
BIN_CM, MIN_COUNT = 2.0, 30
INK = "#333333"
GRID = dict(color="0.85", linewidth=0.6)


# %%
# --- per-dataset machinery ----------------------------------------------------

def raw_context(info):
    """Raw-recording arrays + cohort threshold for one dataset."""
    mat = load_mat(mat_path(ROOT / "data" / "raw", info))
    return {
        "pos": np.asarray(mat["position"], dtype=float).ravel(),
        "vel": np.asarray(mat["velocity"], dtype=float).ravel(),
        "tfb": np.asarray(mat["trialFinalBin"], dtype=int).ravel() - 1,
        "dprime": np.asarray(mat["dPrime"], dtype=float).ravel(),
        "thresh": 2 if info.cohort == "expert" else -1.1,
    }


def kept_raw_indices(ctx, bins_before, bins_after):
    """Raw-bin indices surviving the row filters for this window size (library logic)."""
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


def load_run(info, ctx, cfg):
    study = info.study
    pkl = pickle_path(ROOT / cfg["data"], info, "_withCtxt_onlyPosition_flat.pickle")
    with pkl.open("rb") as f:
        _, y, trial_markers = pickle.load(f)  # noqa: S301 — our own preprocess artefact
    kept = kept_raw_indices(ctx, *cfg["window"])
    if len(kept) != len(y) or not np.allclose(ctx["pos"][kept], y[:, 0]):
        raise ValueError(f"{info.display} / {cfg['results']}: kept set does not match the pickle")
    rows = pooled_test_rows(ROOT / cfg["results"], study, y[:, 0], trial_markers)
    final = load_final_results(ROOT / cfg["results"], study)
    folds = [(t * CM_PER_DEGREE, p * CM_PER_DEGREE)
             for t, p in fold_predictions(final, "position")]
    y_true, y_pred, fold_id = pool_folds(folds)
    return {"y_true": y_true, "y_pred": y_pred, "fold_id": fold_id,
            "raw_idx": kept[rows],
            "trials": fold_test_trials(ROOT / cfg["results"], study), **cfg}


def load_dataset(info):
    ctx = raw_context(info)
    runs = {name: load_run(info, ctx, cfg) for name, cfg in RUNS.items()}
    ref = runs["baseline raw·2/1/2"]["trials"]
    for name, d in runs.items():
        assert d["trials"] == ref, f"{info.display} / {name}: fold partition differs"
    kept22 = kept_raw_indices(ctx, 2, 2)
    for d in runs.values():
        d["common"] = np.isin(d["raw_idx"], kept22)
    return runs


# %%
# --- discover the datasets present in all four trees --------------------------

print("=" * 100)
print("ITEM 7 — 2×2 smoothing × window (common support, paired folds)")
print("=" * 100)
per_tree = [{i.display: i for i in discover_studies(ROOT / cfg["results"], "pos")}
            for cfg in RUNS.values()]
common_names = sorted(set.intersection(*[set(t) for t in per_tree]))
print(f"datasets in all four trees: {common_names}")

datasets = {}
for name in common_names:
    info = per_tree[0][name]
    datasets[name] = load_dataset(info)
    n_common = int(datasets[name]["baseline raw·2/1/2"]["common"].sum())
    print(f"  ✅ {name}: common support {n_common:,} bins, folds paired")


# %%
# --- per-fold metrics on the common support ----------------------------------

rows = []
for ds_name, runs in datasets.items():
    for run_name, d in runs.items():
        for k in np.unique(d["fold_id"]):
            m = (d["fold_id"] == k) & d["common"]
            rows.append({"Dataset": ds_name, "run": run_name, "fold": int(k),
                         **error_summary(d["y_true"], d["y_pred"], m)})
per_fold = pd.DataFrame(rows)
per_fold.to_csv(PLOT_DIR / "per_fold_metrics.csv", index=False, float_format="%.4f")

med = per_fold.groupby(["Dataset", "run"], sort=False)[["n", "r2", "rmse", "mae"]].median()


def paired(ds, run_a, run_b, metric):
    sub = per_fold[per_fold.Dataset == ds]
    a = sub[sub.run == run_a].set_index("fold")[metric]
    b = sub[sub.run == run_b].set_index("fold")[metric]
    delta = a - b
    better = (delta > 0) if metric == "r2" else (delta < 0)
    return float(delta.median()), int(better.sum()), len(delta)


def fmt_pair(ds, run_a, run_b, metric, nd):
    d, w, n = paired(ds, run_a, run_b, metric)
    return f"{d:+.{nd}f} ({w}/{n})"


# %%
# --- tables -------------------------------------------------------------------

def write_md(path):
    lines = [
        "# Item 7 — smoothing × window (common support, paired folds)",
        "",
        "All four runs of each dataset evaluated on the same bins (its 2/1/2 kept set). "
        "Per-fold medians, 5 paired folds. MSE = RMSE².",
    ]
    for ds in common_names:
        lines += ["", f"## {ds}", "", "| run | n bins | R² | RMSE | MAE | MSE |",
                  "|---|---:|---:|---:|---:|---:|"]
        for run_name in RUNS:
            r = med.loc[(ds, run_name)]
            lines.append(f"| {run_name} | {int(r['n']):,} | {r['r2']:.3f} | {r['rmse']:.1f} "
                         f"| {r['mae']:.1f} | {r['rmse'] ** 2:.0f} |")
        lines += ["", "| contrast | Δ R² | Δ RMSE | Δ MAE |", "|---|---:|---:|---:|"]
        for a, b, label in CONTRASTS:
            lines.append(f"| {label} | {fmt_pair(ds, a, b, 'r2', 3)} "
                         f"| {fmt_pair(ds, a, b, 'rmse', 1)} | {fmt_pair(ds, a, b, 'mae', 1)} |")
    lines += ["", "## Aggregate — median across datasets (folds won summed)", "",
              "| contrast | Δ R² | Δ RMSE | Δ MAE |", "|---|---:|---:|---:|"]
    for a, b, label in CONTRASTS:
        cells = []
        for metric, nd in (("r2", 3), ("rmse", 1), ("mae", 1)):
            ds_stats = [paired(ds, a, b, metric) for ds in common_names]
            d_med = float(np.median([d for d, _, _ in ds_stats]))
            won = sum(w for _, w, _ in ds_stats)
            tot = sum(n for _, _, n in ds_stats)
            cells.append(f"{d_med:+.{nd}f} ({won}/{tot})")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    Path(path).write_text("\n".join(lines) + "\n")


write_md(PLOT_DIR / "comparison_table.md")
med.reset_index().to_csv(PLOT_DIR / "comparison_table.csv", index=False, float_format="%.4f")
print(f"📁 {PLOT_DIR / 'comparison_table.md'}")


# %%
# --- MAE-by-position overlay per dataset --------------------------------------

def overlay(ds_name, runs):
    base = runs["baseline raw·2/1/2"]
    hi = float(np.ceil(np.percentile(base["y_true"], 99.9) / 10) * 10)
    edges = np.arange(0, hi + BIN_CM, BIN_CM)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    for run_name, d in runs.items():
        tbl = binned_error(d["y_true"][d["common"]], d["y_pred"][d["common"]],
                           edges, min_count=MIN_COUNT)
        ax.plot(tbl["center"], tbl["mae"], color=d["color"], linewidth=1.7, label=run_name)
    ax.axvspan(CONTEXT_START_CM, CONTEXT_END_CM, color="gold", alpha=0.15, lw=0)
    ax.axvline(CONTEXT_END_CM, color="0.35", linewidth=0.8, linestyle=":")
    ax.set_xlabel("True position (cm)", fontsize=10, color=INK)
    ax.set_ylabel("MAE (cm)", fontsize=10, color=INK)
    ax.set_title(f"{ds_name} — MAE by position, four runs on the common support · "
                 "test predictions, 5 folds pooled", fontsize=10.5, color=INK, loc="left")
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    out = PLOT_DIR / f"mae_by_position_overlay_{ds_name}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"📁 {out}")


for ds_name, runs in datasets.items():
    overlay(ds_name, runs)


# %%
# --- headline -----------------------------------------------------------------

print()
print("=" * 100)
print("HEADLINE — per-fold medians on the common support")
print("=" * 100)
for ds in common_names:
    parts = [f"{run.split(' ')[0]} R² {med.loc[(ds, run)]['r2']:.3f}" for run in RUNS]
    print(f"  {ds:<16} " + " · ".join(parts))
print()
for a, b, label in CONTRASTS:
    ds_stats = [paired(ds, a, b, "r2") for ds in common_names]
    d_med = float(np.median([d for d, _, _ in ds_stats]))
    won = sum(w for _, w, _ in ds_stats)
    tot = sum(n for _, _, n in ds_stats)
    print(f"  {label:<32} ΔR² median {d_med:+.3f} · folds won {won}/{tot}")
