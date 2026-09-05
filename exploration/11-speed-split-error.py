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
# # Is speed a confound for the position decoder? Slow vs fast test bins
#
# First pass of the position/velocity dissociation analysis (Phase 5, Item 5;
# operational definition from the director, 2026-09-05): split every test bin
# by the animal's running speed at that bin — **slow** = below the dataset's
# median speed over the surviving bins, **fast** = above — and compare the
# position-decoding error between the halves. If the decoder reads position
# rather than reconstructing it from speed, the error should not depend much
# on the speed regime.
#
# Speed and position are correlated (the animal is slow at the trial start and
# at the reward zone), so the split is also reported **stratified by corridor
# zone** — without that, "slow vs fast" is partly "corridor ends vs middle"
# and the comparison confounds itself.
#
# Method: pooled test predictions mapped back to pickle rows
# (`analysis.results.pooled_test_rows`, exact-verified), speed taken from the
# `_withCtxt_flat` pickle's second target column (cm/s), same rows.
#
# Outputs under `results/plots/single/speed-split/`: per-dataset figure
# (speed histogram + MAE slow/fast per zone), `summary_table.csv/.md`,
# per-fold metrics, stdout headline.
#
# Env overrides: `RESULTS_DIR`, `DATA_DIR`, `PLOT_DIR`.

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

from analysis.loaders import CM_PER_DEGREE, CONTEXT_END_CM, CONTEXT_START_CM  # noqa: E402
from analysis.metrics import error_summary  # noqa: E402
from analysis.results import (  # noqa: E402
    discover_studies,
    fold_predictions,
    load_final_results,
    pickle_path,
    pool_folds,
    pooled_test_rows,
)

# %%
# --- config -----------------------------------------------------------------

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "processed"))
PLOT_DIR = Path(os.environ.get("PLOT_DIR", RESULTS_DIR / "plots" / "single" / "speed-split"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ZONES = {
    "pre_context": (-np.inf, CONTEXT_START_CM),
    "context": (CONTEXT_START_CM, CONTEXT_END_CM),
    "post_context": (CONTEXT_END_CM, np.inf),
}
SPEED_COLOR = {"slow": "#7E57C2", "fast": "#E76F51"}
INK = "#333333"
GRID = dict(color="0.85", linewidth=0.6)


# %%
# --- load studies with per-bin speed -----------------------------------------

def load_speed_study(info):
    final = load_final_results(RESULTS_DIR, info.study)
    folds = [(t * CM_PER_DEGREE, p * CM_PER_DEGREE)
             for t, p in fold_predictions(final, "position")]
    y_true, y_pred, fold_id = pool_folds(folds)
    with pickle_path(DATA_DIR, info).open("rb") as f:
        _, y, trial_markers = pickle.load(f)  # noqa: S301 — our own preprocess artefact
    rows = pooled_test_rows(RESULTS_DIR, info.study, y[:, 0], trial_markers)
    speed = y[rows, 1]
    threshold = float(np.median(speed))
    return {"info": info, "y_true": y_true, "y_pred": y_pred, "fold_id": fold_id,
            "speed": speed, "threshold": threshold}


print("=" * 100)
print(f"SPEED SPLIT — studies under {RESULTS_DIR}")
print("=" * 100)
studies = {}
for info in discover_studies(RESULTS_DIR, "pos"):
    try:
        studies[info.display] = load_speed_study(info)
        d = studies[info.display]
        print(f"  ✅ {info.display}: median speed {d['threshold']:.1f} cm/s")
    except FileNotFoundError as e:
        print(f"  ⚠️  {info.display}: {e} — skipped")


# %%
# --- metrics: slow vs fast, overall and per zone ------------------------------

def split_rows(name, d):
    slow = d["speed"] < d["threshold"]
    rows = []
    for regime, m in (("slow", slow), ("fast", ~slow)):
        row = {"Dataset": name, "regime": regime,
               "speed_threshold": d["threshold"],
               **{f"{k}_all": v for k, v in
                  error_summary(d["y_true"], d["y_pred"], m).items()}}
        for zone, (lo, hi) in ZONES.items():
            zm = m & (d["y_true"] >= lo) & (d["y_true"] < hi)
            zs = error_summary(d["y_true"], d["y_pred"], zm)
            row[f"mae_{zone}"] = zs["mae"]
            row[f"rmse_{zone}"] = zs["rmse"]
            row[f"n_{zone}"] = zs["n"]
        rows.append(row)
    return rows


summary = pd.DataFrame([r for name, d in studies.items() for r in split_rows(name, d)])
csv_path = PLOT_DIR / "summary_table.csv"
summary.to_csv(csv_path, index=False, float_format="%.4f")
print(f"📁 {csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path}")


# %%
# --- per-dataset figure: speed histogram + per-zone MAE slow/fast -------------

def style_axes(ax):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)


def plot_speed_hist(ax, d):
    ax.hist(d["speed"], bins=60, color="#7B848F", alpha=0.8)
    ax.axvline(d["threshold"], color="0.2", linewidth=1.2, linestyle="--",
               label=f"median = {d['threshold']:.1f} cm/s")
    ax.set_xlabel("Speed (cm/s)", fontsize=10, color=INK)
    ax.set_ylabel("Test bins", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    style_axes(ax)


def plot_zone_mae(ax, name):
    sub = summary[summary.Dataset == name].set_index("regime")
    zones = ["mae_all"] + [f"mae_{z}" for z in ZONES]
    labels = ["all bins", "pre-context", "context", "post-context"]
    x = np.arange(len(zones))
    for k, regime in enumerate(("slow", "fast")):
        vals = [sub.loc[regime, c.replace("mae_all", "mae_all")] for c in zones]
        ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=SPEED_COLOR[regime],
               label=regime)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylabel("MAE (cm)", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)


def figure_for_dataset(name, d):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    plot_speed_hist(axes[0], d)
    plot_zone_mae(axes[1], name)
    fig.suptitle(f"{name} — position error by speed regime · test predictions, "
                 "5 outer folds pooled", fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    out = PLOT_DIR / f"{name}_speed_split.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"📁 {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


for name, d in studies.items():
    figure_for_dataset(name, d)


# %%
# --- markdown twin ------------------------------------------------------------

def fmt(x, nd=1):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


def write_md(t, path):
    lines = [
        "# Position error by speed regime (slow / fast, median split per dataset)",
        "",
        "Generated by `exploration/11-speed-split-error.py`. Slow = below the dataset's "
        "median test-bin speed. Zones stratify by true position because speed and "
        "position are correlated (slow at the start and at the reward zone) — compare "
        "slow vs fast **within** a zone, not only overall. MSE = RMSE².",
        "",
        "| Dataset | regime | n | R² | RMSE | MAE | MAE pre | MAE ctx | MAE post |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['Dataset']} | {r['regime']} | {int(r['n_all']):,} | {fmt(r['r2_all'], 3)} "
            f"| {fmt(r['rmse_all'])} | {fmt(r['mae_all'])} "
            f"| {fmt(r['mae_pre_context'])} | {fmt(r['mae_context'])} "
            f"| {fmt(r['mae_post_context'])} |")
    Path(path).write_text("\n".join(lines) + "\n")


md_path = PLOT_DIR / "summary_table.md"
write_md(summary, md_path)
print(f"📁 {md_path.relative_to(ROOT) if md_path.is_relative_to(ROOT) else md_path}")


# %%
# --- headline -----------------------------------------------------------------

piv = summary.pivot(index="Dataset", columns="regime",
                    values=["mae_all", "rmse_all", "r2_all",
                            "mae_pre_context", "mae_context", "mae_post_context"])
print()
print("=" * 100)
print("HEADLINE — medians across datasets (slow vs fast)")
print("=" * 100)
for metric, label in (("mae_all", "MAE all bins"), ("rmse_all", "RMSE all bins"),
                      ("r2_all", "R² all bins"), ("mae_pre_context", "MAE pre-context"),
                      ("mae_context", "MAE context"), ("mae_post_context", "MAE post-context")):
    slow = piv[(metric, "slow")].median()
    fast = piv[(metric, "fast")].median()
    print(f"  {label:<18} slow {slow:6.2f} · fast {fast:6.2f} · Δ(slow−fast) {slow - fast:+6.2f}")
