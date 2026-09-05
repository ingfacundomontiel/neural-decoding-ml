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
# # Position decoding error along the corridor — real vs predicted
#
# Where along the virtual corridor does the position decoder fail? For every
# single-output position study (`results/mlp_<dataset>_bins200ms_pos/`) we pool
# the five outer test folds (disjoint trial partitions → one out-of-sample
# prediction per surviving bin), convert to cm, and look at the error as a
# function of the animal's **true** position.
#
# Motivation: linear baselines run by the lab suggested the errors concentrate at
# the two ends of the corridor — the start (just after the teleport) and the
# stretch past the context exit (reward zone / licking drift). This notebook
# checks the same thing on the thesis MLPs and produces the figures + tables that
# motivate the position-tail filter experiment (Phase 5, Item 6).
#
# Per dataset, one figure with three rows sharing the x axis (true position, cm):
#
# 1. **Conditional heatmap** — true (x) vs predicted (y). Each 2 cm column is the
#    distribution of the predictions for that true position, **normalised by the
#    column's own sample count** (P(predicted | true)), so densely and sparsely
#    visited positions read on the same scale. Columns with fewer than 30 test
#    bins are blanked. Perfect decoding sits on the identity line.
# 2. **Error vs true position** — median of (pred − true) per 2 cm bin with the
#    IQR band (positive = over-prediction), plus MAE and RMSE per bin (grey), which
#    shows where the error *concentrates* regardless of sign. Each zone is
#    annotated with its share of the test bins and its share of the total
#    squared error: a zone whose error share exceeds its bin share concentrates
#    error.
# 3. **Support** — how many test bins fall in each 2 cm bin (sparse positions →
#    noisy estimates).
#
# The context zone (44.7 → 62.9 cm) is outlined on the identity panel (edges only,
# so the density colours stay readable) and shaded on the two line panels.
#
# Emits under `results/plots/single/error-by-position/`:
#
# - `<dataset>_error_by_position.svg` — one per position study.
# - `hero_experts_error_by_position.svg` — 3 columns: Pir N31, CA3 S18, DG S17.
# - `hero_piriform_experts_error_by_position.svg` — 3 columns: Pir N31, L3, L5.
# - `summary_table.csv` / `.md` — full-support vs. ≤ context-exit metrics, per-zone MAE.
# - `per_fold_metrics.csv` — RMSE / MAE / R² per outer fold, full and ≤ context-exit.
#
# Env overrides (so the same script serves later batches):
# `RESULTS_DIR` (default `results`), `PLOT_DIR` (default
# `<RESULTS_DIR>/plots/single/error-by-position`).

# %%
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no GUI window per figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from analysis.loaders import CM_PER_DEGREE, CONTEXT_END_CM, CONTEXT_START_CM  # noqa: E402
from analysis.metrics import (  # noqa: E402
    binned_error,
    error_summary,
    per_fold_summary,
    zone_summary,
)
from analysis.results import (  # noqa: E402
    discover_studies,
    fold_predictions,
    fold_r2_scores,
    load_final_results,
    pool_folds,
)

# %%
# --- config -----------------------------------------------------------------

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
PLOT_DIR = Path(os.environ.get("PLOT_DIR", RESULTS_DIR / "plots" / "single" / "error-by-position"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "pos"
HERO = ["Pir-Expert-N31", "CA3-Expert-S18", "DG-Expert-S17"]
HERO_PIRIFORM = ["Pir-Expert-N31", "Pir-Expert-L3", "Pir-Expert-L5"]

BIN_CM = 2.0        # width of the true-position bins for the error curve
MIN_COUNT = 30      # bins with fewer test samples get no error estimate
AXIS_LO_CM = -10.0  # identity plot lower limit (predictions can undershoot 0)

# Type-level colour scheme shared with 03/06/07.
COLOR_SCHEME = {
    'pir_naive':   "#58AAE9",
    'pir_experto': "#004E89",
    'CA3_naive':   "#69DA85",
    'CA3_experto': "#146D02",
    'DG_naive':    "#EBA447",
    'DG_experto':  "#9B4615",
}
CONTEXT_FILL = "gold"
INK = "#333333"
GRID = dict(color="0.85", linewidth=0.6)

FIXED_ORDER = [
    'Pir-Naive-N53', 'Pir-Naive-L7',
    'Pir-Expert-N31', 'Pir-Expert-L3', 'Pir-Expert-L5', 'Pir-Expert-N59',
    'CA3-Naive-S30', 'CA3-Naive-S31',
    'CA3-Expert-S17', 'CA3-Expert-S18',
    'DG-Naive-S24', 'DG-Naive-S38', 'DG-Naive-S39', 'DG-Naive-S41',
    'DG-Expert-S17', 'DG-Expert-S18', 'DG-Expert-S19',
]

ZONES = {
    "pre_context": (-np.inf, CONTEXT_START_CM),
    "context": (CONTEXT_START_CM, CONTEXT_END_CM),
    "post_context": (CONTEXT_END_CM, np.inf),
}


def within_context_exit(y_true_cm):
    """Evaluation mask shared by every 'restricted support' number below."""
    return y_true_cm <= CONTEXT_END_CM


# %%
# --- load every position study, pooled and in cm ----------------------------

def load_study(info):
    final = load_final_results(RESULTS_DIR, info.study)
    folds = [(t * CM_PER_DEGREE, p * CM_PER_DEGREE) for t, p in fold_predictions(final, "position")]
    y_true, y_pred, fold_id = pool_folds(folds)
    return {"info": info, "folds": folds, "y_true": y_true, "y_pred": y_pred,
            "fold_id": fold_id, "r2_folds": fold_r2_scores(final, "position")}


print("=" * 100)
print(f"POSITION ERROR ALONG THE CORRIDOR — studies under {RESULTS_DIR}")
print("=" * 100)

studies = {info.display: load_study(info) for info in discover_studies(RESULTS_DIR, TARGET)}
print(f"Position studies found: {len(studies)}")
for name in [d for d in FIXED_ORDER if d in studies]:
    d = studies[name]
    print(f"  {name:<16} {len(d['y_true']):>6} test bins   "
          f"fold-median R² = {np.median(d['r2_folds']):.3f}")


# %%
# --- plotting helpers -------------------------------------------------------

def _mix(color, other, w):
    """Linear mix of two colours in RGB; w = weight of *other*."""
    c, o = np.array(to_rgb(color)), np.array(to_rgb(other))
    return tuple((1 - w) * c + w * o)


def single_hue_cmap(color):
    """Light → dark ramp of one hue (density colouring for the hexbin)."""
    return LinearSegmentedColormap.from_list(
        f"ramp_{color}", [_mix(color, "white", 0.82), color, _mix(color, "black", 0.35)])


def axis_hi(y_true):
    """Upper axis limit: a round number just above the 99.9th percentile of true position."""
    hi = max(np.percentile(y_true, 99.9), CONTEXT_END_CM + 10)
    return float(np.ceil(hi / 10) * 10)


def shade_context(ax, axis="x", label=None):
    span = ax.axvspan if axis == "x" else ax.axhspan
    span(CONTEXT_START_CM, CONTEXT_END_CM, color=CONTEXT_FILL, alpha=0.15, lw=0, label=label)
    line = ax.axvline if axis == "x" else ax.axhline
    line(CONTEXT_END_CM, color="0.35", linewidth=0.8, linestyle=":")


# Fine dots so the context edges read differently from the dashed identity line.
CONTEXT_EDGE = dict(color="0.35", linewidth=1.0, linestyle=(0, (1.2, 2.2)))


def outline_context(ax):
    """Context-zone edges on both axes (no fill — keeps the hexbin colours readable)."""
    for v in (CONTEXT_START_CM, CONTEXT_END_CM):
        ax.axvline(v, **CONTEXT_EDGE)
        ax.axhline(v, **CONTEXT_EDGE)


def context_handle():
    return Line2D([], [], label=f"context zone ({CONTEXT_START_CM:g}–{CONTEXT_END_CM:g} cm)",
                  **CONTEXT_EDGE)


def style_axes(ax):
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)


def plot_identity(ax, y_true, y_pred, color, hi):
    """Column-normalised heatmap of predicted vs true position (identity panel).

    Each true-position column is divided by its own sample count, so the colour
    shows P(predicted | true) rather than joint density — the support histogram
    (bottom row) carries the sample counts the normalisation removes here.
    """
    lo = AXIS_LO_CM
    inside = (y_pred >= lo) & (y_pred <= hi)
    x_edges = np.arange(0.0, hi + BIN_CM, BIN_CM)   # true positions are >= 0
    y_edges = np.arange(lo, hi + BIN_CM, BIN_CM)
    counts, _, _ = np.histogram2d(y_true, y_pred, bins=[x_edges, y_edges])
    col_total = np.histogram(y_true, bins=x_edges)[0].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = counts / col_total[:, None]
    cond[col_total < MIN_COUNT, :] = np.nan  # no distribution from a handful of bins
    cond[cond == 0] = np.nan                 # empty cells stay background-white
    vmax = float(np.nanpercentile(cond, 99)) if np.isfinite(cond).any() else None
    ax.pcolormesh(x_edges, y_edges, cond.T, cmap=single_hue_cmap(color),
                  vmin=0.0, vmax=vmax, shading="flat")
    ax.plot([lo, hi], [lo, hi], color="0.25", linewidth=1.0, linestyle="--", label="identity")
    outline_context(ax)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylabel("Predicted position (cm)", fontsize=10, color=INK)
    if (~inside).any():
        ax.text(0.02, 0.97, f"{100 * (~inside).mean():.1f}% of predictions outside frame",
                transform=ax.transAxes, fontsize=7.5, color="0.4", va="top")
    style_axes(ax)


def plot_binned_error(ax, table, color, zones=None, hi=None):
    """Median signed error ± IQR and MAE per true-position bin, with per-zone error shares."""
    ax.fill_between(table["center"], table["q25"], table["q75"], color=color, alpha=0.22, lw=0,
                    label="IQR")
    ax.plot(table["center"], table["median_signed"], color=color, linewidth=1.8,
            label="median (pred − true)")
    ax.plot(table["center"], table["mae"], color="0.3", linewidth=1.3, label="MAE")
    ax.plot(table["center"], table["rmse"], color="0.3", linewidth=1.1,
            linestyle=(0, (4, 2)), label="RMSE")
    ax.axhline(0, color="0.25", linewidth=0.9)
    shade_context(ax, "x")
    if zones is not None:
        annotate_zone_shares(ax, zones, hi)
    ax.set_ylabel("Error (cm)", fontsize=10, color=INK)
    style_axes(ax)


def annotate_zone_shares(ax, zones, hi):
    """Print "<bins> % of bins / <error> % of sq. error" centred in each zone, top of the axes."""
    centres = {
        "pre_context": 0.5 * (0 + CONTEXT_START_CM),
        "context": 0.5 * (CONTEXT_START_CM + CONTEXT_END_CM),
        "post_context": 0.5 * (CONTEXT_END_CM + hi),
    }
    for _, row in zones.iterrows():
        if row["pct_bins"] < 0.5:  # empty zone (e.g. post-context in the tail-filtered variant)
            continue
        ax.text(centres[row["zone"]], 0.97,
                f"{row['pct_bins']:.0f}% of bins\n{row['pct_sq_error']:.0f}% of sq. error",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=7.5, color="0.35", linespacing=1.15)


def plot_support(ax, table, color):
    """Number of test bins per true-position bin."""
    ax.bar(table["center"], table["n"], width=BIN_CM * 0.9, color=_mix(color, "white", 0.35), lw=0)
    ax.axhline(MIN_COUNT, color="0.5", linewidth=0.7, linestyle=":")
    shade_context(ax, "x")
    ax.set_ylabel("Test bins", fontsize=10, color=INK)
    ax.set_xlabel("True position (cm)", fontsize=10, color=INK)
    style_axes(ax)


def panel_title(name, d):
    full = error_summary(d["y_true"], d["y_pred"])
    return (f"{name}\n"
            f"fold-median R² = {np.median(d['r2_folds']):.3f} · "
            f"MAE = {full['mae']:.1f} cm · RMSE = {full['rmse']:.1f} cm · n = {full['n']:,}")


def draw_column(axes, name, d, hi=None):
    """Fill a (3,) column of axes with the three rows for one study.

    *hi* fixes the upper axis limit (cm); default = per-dataset :func:`axis_hi`.
    """
    color = COLOR_SCHEME.get(d["info"].dtype, "#333333")
    hi = axis_hi(d["y_true"]) if hi is None else hi
    edges = np.arange(0, hi + BIN_CM, BIN_CM)
    table = binned_error(d["y_true"], d["y_pred"], edges, min_count=MIN_COUNT)
    zones = zone_summary(d["y_true"], d["y_pred"], ZONES)
    plot_identity(axes[0], d["y_true"], d["y_pred"], color, hi)
    plot_binned_error(axes[1], table, color, zones=zones, hi=hi)
    plot_support(axes[2], table, color)
    for ax in axes[1:]:
        ax.set_xlim(AXIS_LO_CM, hi)
    y0, y1 = axes[1].get_ylim()
    axes[1].set_ylim(y0, y1 + 0.28 * (y1 - y0))  # head-room for the zone annotations
    axes[0].set_title(panel_title(name, d), fontsize=10.5, color=INK, loc="left", pad=8)
    return table


def figure_for_dataset(name, d):
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 12), dpi=200,
                             gridspec_kw={"height_ratios": [3, 1.6, 0.9]}, sharex=False)
    table = draw_column(axes, name, d)
    handles, _ = axes[0].get_legend_handles_labels()
    axes[0].legend(handles=handles + [context_handle()], loc="lower right", fontsize=8,
                   frameon=False)
    axes[1].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Position decoding error along the corridor — single-output MLP · "
                 "test predictions, 5 outer folds pooled · top: P(predicted | true)",
                 fontsize=11, color=INK, x=0.02, ha="left", y=0.995)
    fig.tight_layout()
    out = PLOT_DIR / f"{name}_error_by_position.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"📁 {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return table


def hero_figure(names, out_name, subtitle):
    present = [n for n in names if n in studies]
    if not present:
        print("❌ none of the hero datasets found — skipping hero figure")
        return
    fig, axes = plt.subplots(3, len(present), figsize=(6.2 * len(present), 12), dpi=200,
                             gridspec_kw={"height_ratios": [3, 1.6, 0.9]}, squeeze=False)
    hi = max(axis_hi(studies[n]["y_true"]) for n in present)  # common axes across columns
    for col, name in enumerate(present):
        draw_column(axes[:, col], name, studies[name], hi=hi)
        if col > 0:
            for ax in axes[:, col]:
                ax.set_ylabel("")
    handles, _ = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles=handles + [context_handle()], loc="lower right", fontsize=8,
                      frameon=False)
    axes[1, 0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle(f"Position decoding error along the corridor — {subtitle}, single-output MLP · "
                 "test predictions, 5 outer folds pooled · top: P(predicted | true)",
                 fontsize=12, color=INK, x=0.02, ha="left", y=0.995)
    fig.tight_layout()
    out = PLOT_DIR / out_name
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"📁 {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


# %%
# --- figures ----------------------------------------------------------------

binned_tables = {}
for name in [d for d in FIXED_ORDER if d in studies]:
    binned_tables[name] = figure_for_dataset(name, studies[name])
hero_figure(HERO, "hero_experts_error_by_position.svg", "expert animals")
hero_figure(HERO_PIRIFORM, "hero_piriform_experts_error_by_position.svg", "piriform expert animals")


# %%
# --- summary table: full support vs. ≤ context exit, per-zone MAE -----------

def summarise_study(name, d):
    y_true, y_pred = d["y_true"], d["y_pred"]
    full = error_summary(y_true, y_pred)
    restricted = error_summary(y_true, y_pred, within_context_exit(y_true))
    zones = zone_summary(y_true, y_pred, ZONES).set_index("zone")
    row = {
        "Dataset": name, "Type": d["info"].dtype, "n_bins": full["n"],
        "pct_post_context": 100 * (~within_context_exit(y_true)).mean(),
        "r2_fold_median": float(np.median(d["r2_folds"])),
        "r2_pooled": full["r2"], "r2_pooled_ctx": restricted["r2"],
        "rmse_cm": full["rmse"], "rmse_cm_ctx": restricted["rmse"],
        "mae_cm": full["mae"], "mae_cm_ctx": restricted["mae"],
        "chance_rmse_cm": full["chance_rmse"], "chance_rmse_cm_ctx": restricted["chance_rmse"],
    }
    for zone in ZONES:
        row[f"pct_bins_{zone}"] = zones.loc[zone, "pct_bins"]
        row[f"pct_sq_err_{zone}"] = zones.loc[zone, "pct_sq_error"]
        row[f"mae_{zone}"] = zones.loc[zone, "mae"]
        row[f"bias_{zone}"] = zones.loc[zone, "bias"]
    return row


order_map = {d: i for i, d in enumerate(FIXED_ORDER)}
summary = pd.DataFrame([summarise_study(n, d) for n, d in studies.items()])
summary = summary.sort_values("Dataset", key=lambda s: s.map(order_map)).reset_index(drop=True)
csv_path = PLOT_DIR / "summary_table.csv"
summary.to_csv(csv_path, index=False, float_format="%.4f")
print(f"📁 {csv_path.relative_to(ROOT) if csv_path.is_relative_to(ROOT) else csv_path}")


def fmt(x, nd=1):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


def write_summary_md(t, path):
    lines = [
        "# Position decoding error along the corridor — single-output MLP",
        "",
        "Generated by `exploration/08-position-error-analysis.py`. All errors in cm "
        f"(`CM_PER_DEGREE = {CM_PER_DEGREE}`). Five outer test folds pooled per dataset.",
        "",
        f"**'≤ ctx' columns** restrict the evaluation to test bins with true position ≤ "
        f"{CONTEXT_END_CM} cm (context exit). Same predictions, fewer bins — the drop in R² "
        "with a near-unchanged MAE is the support effect described in D-018.",
        "",
        "## Full support vs. ≤ context exit",
        "",
        "| Dataset | n bins | % post-ctx | R² fold-med | R² pooled | R² ≤ ctx | RMSE | RMSE ≤ ctx "
        "| MAE | MAE ≤ ctx | chance RMSE | chance RMSE ≤ ctx |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['Dataset']} | {int(r['n_bins']):,} | {fmt(r['pct_post_context'])} "
            f"| {fmt(r['r2_fold_median'], 3)} | {fmt(r['r2_pooled'], 3)} "
            f"| {fmt(r['r2_pooled_ctx'], 3)} "
            f"| {fmt(r['rmse_cm'])} | {fmt(r['rmse_cm_ctx'])} "
            f"| {fmt(r['mae_cm'])} | {fmt(r['mae_cm_ctx'])} "
            f"| {fmt(r['chance_rmse_cm'])} | {fmt(r['chance_rmse_cm_ctx'])} |")
    lines += [
        "",
        f"## Per zone (true position): pre-context < {CONTEXT_START_CM} cm, "
        f"context {CONTEXT_START_CM}–{CONTEXT_END_CM} cm, post-context > {CONTEXT_END_CM} cm",
        "",
        "Bias = mean(pred − true); positive = over-prediction. `% bins` / `% err²` = the zone's "
        "share of the test bins / of the total squared error — a zone whose error share exceeds "
        "its bin share concentrates error.",
        "",
        "| Dataset | % bins pre / ctx / post | % err² pre / ctx / post "
        "| MAE pre | bias pre | MAE ctx | bias ctx | MAE post | bias post |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t.iterrows():
        shares_b = " / ".join(fmt(r[f"pct_bins_{z}"], 0) for z in ZONES)
        shares_e = " / ".join(fmt(r[f"pct_sq_err_{z}"], 0) for z in ZONES)
        lines.append(
            f"| {r['Dataset']} | {shares_b} | {shares_e} "
            f"| {fmt(r['mae_pre_context'])} | {fmt(r['bias_pre_context'], 1)} "
            f"| {fmt(r['mae_context'])} | {fmt(r['bias_context'], 1)} "
            f"| {fmt(r['mae_post_context'])} | {fmt(r['bias_post_context'], 1)} |")
    Path(path).write_text("\n".join(lines) + "\n")


md_path = PLOT_DIR / "summary_table.md"
write_summary_md(summary, md_path)
print(f"📁 {md_path.relative_to(ROOT) if md_path.is_relative_to(ROOT) else md_path}")


# %%
# --- per-fold metrics (feeds boxplots / cross-batch comparisons later) ------

fold_rows = []
for name, d in studies.items():
    full = per_fold_summary(d["folds"]).assign(support="full")
    ctx = per_fold_summary(d["folds"], mask_fn=within_context_exit)
    ctx = ctx.assign(support="le_context_exit")
    fold_rows.append(pd.concat([full, ctx]).assign(Dataset=name, Type=d["info"].dtype))
per_fold = pd.concat(fold_rows, ignore_index=True)
per_fold = per_fold[["Dataset", "Type", "support", "fold", "n", "r2", "rmse", "mae",
                     "median_ae", "bias", "chance_rmse"]]
pf_path = PLOT_DIR / "per_fold_metrics.csv"
per_fold.to_csv(pf_path, index=False, float_format="%.4f")
print(f"📁 {pf_path.relative_to(ROOT) if pf_path.is_relative_to(ROOT) else pf_path}")


# %%
# --- headline ---------------------------------------------------------------

print()
print("=" * 100)
print("HEADLINE — medians across datasets")
print("=" * 100)
print(f"  % of test bins past the context exit : {summary['pct_post_context'].median():.1f}%")
print(f"  R² pooled  full → ≤ ctx              : {summary['r2_pooled'].median():+.3f} → "
      f"{summary['r2_pooled_ctx'].median():+.3f}")
print(f"  MAE (cm)   full → ≤ ctx              : {summary['mae_cm'].median():.1f} → "
      f"{summary['mae_cm_ctx'].median():.1f}")
print(f"  bias (cm)  pre / ctx / post          : {summary['bias_pre_context'].median():+.1f} / "
      f"{summary['bias_context'].median():+.1f} / {summary['bias_post_context'].median():+.1f}")
print(f"  MAE (cm)   pre / ctx / post          : {summary['mae_pre_context'].median():.1f} / "
      f"{summary['mae_context'].median():.1f} / {summary['mae_post_context'].median():.1f}")
for label, key in (("% bins    ", "pct_bins"), ("% sq.error", "pct_sq_err")):
    shares = " / ".join(f"{summary[f'{key}_{z}'].median():.0f}" for z in ZONES)
    print(f"  {label} pre / ctx / post          : {shares}")
