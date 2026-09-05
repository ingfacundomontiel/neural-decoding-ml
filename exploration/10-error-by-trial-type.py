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
# # Position decoding error by trial type
#
# Hypothesis (director, 2026-09-05): the large error past the context exit is
# driven by the **rewarded trials in which the animal licks** — the rig extends
# the corridor while licking (drift), so those trials own the deep tail and the
# reward-consumption epoch. If so, the error-by-position curve should blow up
# for that trial type and stay tame for the others.
#
# Method: pool the five outer test folds of every single-output position study
# (`results/mlp_*_pos/`), map each pooled prediction back to its pickle row
# (`analysis.results.pooled_test_rows`, verified exactly against the stored
# targets), read the trial id from the pickle's markers, and label each bin with
# its trial's `rewCtxt` / `rewOdor` / `goResponse` from the raw `.mat`
# (per-trial constants, see `data/raw/README.md`). Six groups:
#
# - the four types **ctxt × odor** (reward is delivered only in ctxt+/odor+);
# - the rewarded type split by lick: **ctxt+/odor+ · licked** (drift trials)
#   and **ctxt+/odor+ · no lick**.
#
# Per dataset, one figure: (1) median signed error vs true position for the four
# types; (2) the rewarded split (licked vs no lick) against everything else;
# (3) test-bin support per type. Plus `summary_table.csv/.md` (per group ×
# corridor zone) and a stdout headline (who owns the post-context bins).
#
# Env overrides: `RESULTS_DIR` (default `results`), `DATA_DIR`
# (default `data/processed`), `RAW_DIR` (default `data/raw`), `PLOT_DIR`.

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

from analysis.loaders import CM_PER_DEGREE, CONTEXT_END_CM, CONTEXT_START_CM, load_mat  # noqa: E402
from analysis.metrics import binned_error, error_summary  # noqa: E402
from analysis.results import (  # noqa: E402
    discover_studies,
    fold_predictions,
    load_final_results,
    mat_path,
    pickle_path,
    pool_folds,
    pooled_test_rows,
)

# %%
# --- config -----------------------------------------------------------------

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data" / "processed"))
RAW_DIR = Path(os.environ.get("RAW_DIR", ROOT / "data" / "raw"))
PLOT_DIR = Path(os.environ.get(
    "PLOT_DIR", RESULTS_DIR / "plots" / "single" / "error-by-trial-type"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BIN_CM = 2.0
MIN_COUNT = 20   # per type per position bin — types have fewer samples than the whole set

TYPES = ["ctxt+/odor+", "ctxt+/odor−", "ctxt−/odor+", "ctxt−/odor−"]
TYPE_COLOR = {
    "ctxt+/odor+": "#2A9D8F",   # rewarded
    "ctxt+/odor−": "#E76F51",
    "ctxt−/odor+": "#7E57C2",
    "ctxt−/odor−": "#7B848F",
}
GROUPS = TYPES + ["ctxt+/odor+ · licked", "ctxt+/odor+ · no lick"]
ZONES = {
    "pre_context": (-np.inf, CONTEXT_START_CM),
    "context": (CONTEXT_START_CM, CONTEXT_END_CM),
    "post_context": (CONTEXT_END_CM, np.inf),
}
INK = "#333333"
GRID = dict(color="0.85", linewidth=0.6)


# %%
# --- load every position study with per-bin trial-type labels ----------------

def trial_type_labels(info):
    """Per-trial (ctxt, odor, lick) from the raw .mat, indexed by original trial id."""
    mat = load_mat(mat_path(RAW_DIR, info))
    tfb = np.asarray(mat["trialFinalBin"], dtype=int).ravel() - 1
    return (np.asarray(mat["rewCtxt"]).ravel()[tfb].astype(int),
            np.asarray(mat["rewOdor"]).ravel()[tfb].astype(int),
            np.asarray(mat["goResponse"]).ravel()[tfb].astype(int))


def load_labelled_study(info):
    final = load_final_results(RESULTS_DIR, info.study)
    folds = [(t * CM_PER_DEGREE, p * CM_PER_DEGREE)
             for t, p in fold_predictions(final, "position")]
    y_true, y_pred, _ = pool_folds(folds)
    with pickle_path(DATA_DIR, info).open("rb") as f:
        _, y, trial_markers = pickle.load(f)  # noqa: S301 — our own preprocess artefact
    rows = pooled_test_rows(RESULTS_DIR, info.study, y[:, 0], trial_markers)
    trial = trial_markers[rows]
    ctxt, odor, lick = trial_type_labels(info)
    c, o, li = ctxt[trial], odor[trial], lick[trial]
    type_of_bin = np.where(c == 1,
                           np.where(o == 1, "ctxt+/odor+", "ctxt+/odor−"),
                           np.where(o == 1, "ctxt−/odor+", "ctxt−/odor−"))
    return {"info": info, "y_true": y_true, "y_pred": y_pred,
            "type": type_of_bin, "licked": li == 1}


def group_mask(d, group):
    if group.startswith("ctxt+/odor+ ·"):
        rewarded = d["type"] == "ctxt+/odor+"
        return rewarded & (d["licked"] if "· licked" in group else ~d["licked"])
    return d["type"] == group


print("=" * 100)
print(f"ERROR BY TRIAL TYPE — studies under {RESULTS_DIR}")
print("=" * 100)
studies = {}
for info in discover_studies(RESULTS_DIR, "pos"):
    try:
        studies[info.display] = load_labelled_study(info)
        print(f"  ✅ {info.display}")
    except FileNotFoundError as e:
        print(f"  ⚠️  {info.display}: {e} — skipped")
print(f"Datasets labelled: {len(studies)}")


# %%
# --- figures -----------------------------------------------------------------

def style_axes(ax):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)


def shade_context(ax):
    ax.axvspan(CONTEXT_START_CM, CONTEXT_END_CM, color="gold", alpha=0.15, lw=0)
    ax.axvline(CONTEXT_END_CM, color="0.35", linewidth=0.8, linestyle=":")


def median_curve(d, mask, edges):
    return binned_error(d["y_true"][mask], d["y_pred"][mask], edges, min_count=MIN_COUNT)


def axis_hi(y_true):
    return float(np.ceil(max(np.percentile(y_true, 99.9), CONTEXT_END_CM + 10) / 10) * 10)


def plot_type_medians(ax, d, edges):
    for t in TYPES:
        tbl = median_curve(d, group_mask(d, t), edges)
        ax.plot(tbl["center"], tbl["median_signed"], color=TYPE_COLOR[t], linewidth=1.6,
                label=t)
    ax.axhline(0, color="0.25", linewidth=0.9)
    shade_context(ax)
    ax.set_ylabel("Median signed error (cm)", fontsize=10, color=INK)
    style_axes(ax)


def plot_licked_split(ax, d, edges):
    rewarded = d["type"] == "ctxt+/odor+"
    for label, mask, style in (
        ("ctxt+/odor+ · licked", rewarded & d["licked"], "-"),
        ("ctxt+/odor+ · no lick", rewarded & ~d["licked"], (0, (2, 2))),
    ):
        tbl = median_curve(d, mask, edges)
        ax.plot(tbl["center"], tbl["median_signed"], color=TYPE_COLOR["ctxt+/odor+"],
                linewidth=1.6, linestyle=style,
                label=f"{label} ({int(mask.sum()):,} bins)")
    tbl = median_curve(d, ~rewarded, edges)
    ax.plot(tbl["center"], tbl["median_signed"], color="0.45", linewidth=1.3,
            label=f"all other types ({int((~rewarded).sum()):,} bins)")
    ax.axhline(0, color="0.25", linewidth=0.9)
    shade_context(ax)
    ax.set_ylabel("Median signed error (cm)", fontsize=10, color=INK)
    style_axes(ax)


def plot_type_support(ax, d, edges):
    centers = 0.5 * (edges[:-1] + edges[1:])
    for t in TYPES:
        n, _ = np.histogram(d["y_true"][group_mask(d, t)], bins=edges)
        ax.plot(centers, n, color=TYPE_COLOR[t], linewidth=1.2)
    ax.axhline(MIN_COUNT, color="0.5", linewidth=0.7, linestyle=":")
    shade_context(ax)
    ax.set_ylabel("Test bins", fontsize=10, color=INK)
    ax.set_xlabel("True position (cm)", fontsize=10, color=INK)
    style_axes(ax)


def figure_for_dataset(name, d):
    hi = axis_hi(d["y_true"])
    edges = np.arange(0, hi + BIN_CM, BIN_CM)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10), dpi=200, sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1]})
    plot_type_medians(axes[0], d, edges)
    plot_licked_split(axes[1], d, edges)
    plot_type_support(axes[2], d, edges)
    n_tr = {t: int(group_mask(d, t).sum()) for t in TYPES}
    axes[0].set_title(
        f"{name}\nbins per type: " + " · ".join(f"{t} {n_tr[t]:,}" for t in TYPES),
        fontsize=9.5, color=INK, loc="left", pad=8)
    axes[0].legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
    axes[1].legend(fontsize=8, frameon=False, loc="lower left")
    for ax in axes:
        ax.set_xlim(-5, hi)
    fig.suptitle("Position error by trial type — single-output MLP · "
                 "test predictions, 5 outer folds pooled\n"
                 f"curves blanked where a group has < {MIN_COUNT} bins per {BIN_CM:g} cm",
                 fontsize=10.5, color=INK, x=0.02, ha="left", y=0.998)
    fig.tight_layout()
    out = PLOT_DIR / f"{name}_error_by_trial_type.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"📁 {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


for name, d in studies.items():
    figure_for_dataset(name, d)


# %%
# --- summary table: per group × zone -----------------------------------------

def group_zone_rows(name, d):
    rows = []
    post_total = (d["y_true"] > CONTEXT_END_CM).sum()
    for g in GROUPS:
        m = group_mask(d, g)
        row = {"Dataset": name, "group": g, "n_bins": int(m.sum()),
               "pct_bins": 100 * m.mean()}
        post = m & (d["y_true"] > CONTEXT_END_CM)
        row["pct_of_post_bins"] = 100 * post.sum() / post_total if post_total else np.nan
        for zone, (lo, hi) in ZONES.items():
            zm = m & (d["y_true"] >= lo) & (d["y_true"] < hi)
            zs = error_summary(d["y_true"], d["y_pred"], zm)
            row[f"mae_{zone}"] = zs["mae"]
            row[f"bias_{zone}"] = zs["bias"]
        rows.append(row)
    return rows


summary = pd.DataFrame([r for name, d in studies.items() for r in group_zone_rows(name, d)])
csv_path = PLOT_DIR / "summary_table.csv"
summary.to_csv(csv_path, index=False, float_format="%.4f")
print(f"📁 {csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path}")


def fmt(x, nd=1):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


def write_md(t, path):
    lines = [
        "# Position error by trial type (per corridor zone, cm)",
        "",
        "Generated by `exploration/10-error-by-trial-type.py`. Reward is delivered only in "
        "ctxt+/odor+; the licked/no-lick rows split that type by `goResponse`. "
        "`% of post bins` = share of the post-context test bins owned by the group.",
        "",
        "| Dataset | group | n bins | % bins | % of post bins "
        "| MAE pre | bias pre | MAE ctx | bias ctx | MAE post | bias post |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['Dataset']} | {r['group']} | {int(r['n_bins']):,} | {fmt(r['pct_bins'])} "
            f"| {fmt(r['pct_of_post_bins'])} "
            f"| {fmt(r['mae_pre_context'])} | {fmt(r['bias_pre_context'])} "
            f"| {fmt(r['mae_context'])} | {fmt(r['bias_context'])} "
            f"| {fmt(r['mae_post_context'])} | {fmt(r['bias_post_context'])} |")
    Path(path).write_text("\n".join(lines) + "\n")


md_path = PLOT_DIR / "summary_table.md"
write_md(summary, md_path)
print(f"📁 {md_path.relative_to(ROOT) if md_path.is_relative_to(ROOT) else md_path}")


# %%
# --- headline -----------------------------------------------------------------

print()
print("=" * 100)
print("HEADLINE — medians across datasets")
print("=" * 100)
med = summary.groupby("group", sort=False)[
    ["pct_bins", "pct_of_post_bins", "mae_post_context", "bias_post_context",
     "mae_context", "bias_context"]].median()
for g in GROUPS:
    r = med.loc[g]
    print(f"  {g:<24} {r['pct_bins']:5.1f}% of bins · "
          f"owns {r['pct_of_post_bins']:5.1f}% of post bins"
          f" · post: MAE {r['mae_post_context']:5.1f}, bias {r['bias_post_context']:+6.1f}"
          f" · ctx: MAE {r['mae_context']:5.1f}, bias {r['bias_context']:+6.1f}")
