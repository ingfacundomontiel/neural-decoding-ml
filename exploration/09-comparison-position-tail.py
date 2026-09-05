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
# # Baseline vs. position-tail variant — equal-support comparison
#
# Phase 5 Item 6 (D-017): the variant models were trained on pickles where every
# bin past the context exit (true position > 62.9 cm ≈ 515°, padded `bins_after`)
# was removed. Their test folds therefore only contain positions ≤ 62.9 cm, while
# the baseline's test folds span the whole corridor. Comparing the two on their own
# supports would be meaningless (D-018) — so **both are evaluated on the same bins:
# test bins with true position ≤ 62.9 cm**. For the variant that is (almost) all
# of its bins; for the baseline it is a post-hoc restriction of its stored
# predictions.
#
# Folds are paired: `TrialKFold(n_splits=5, shuffle=True, random_state=42)` splits
# on trial ids, and the variant removes bins but never whole trials, so fold *k*
# holds the same test trials in both runs.
#
# Small caveat: the variant also removes `bins_after` (= 2) bins *before* the
# crossing in every trial, so the baseline's restricted support has ~3 % more
# bins than the variant's (bins at 60–63 cm). Reported alongside.
#
# Inputs (env overrides): `BASELINE_DIR` (default `results`), `VARIANT_DIR`
# (default `results_posend515`), `PLOT_DIR` (default `<VARIANT_DIR>/plots/comparison`).
#
# Emits under `PLOT_DIR`:
#
# - `metrics_paired_boxplot.svg` — RMSE / MAE / R² per fold, baseline≤ctx vs variant,
#   one group per dataset (baseline full support shown greyed as a reference only).
# - `error_by_position_overlay.svg` — MAE and signed error vs true position, both
#   models, one column per dataset.
# - `comparison_table.csv` / `.md` — per-dataset metrics and paired deltas.
# - `compact_table.md` — the same comparison reduced to bins / R² / RMSE / MSE, baseline
#   vs variant, nothing else (the version to show in a meeting).
# - `zone_comparison.csv` / `.md` — MAE and bias per position zone (start → context
#   exit) for both models, pooled folds: shows *where* along the corridor the
#   variant gains or loses, not just the aggregate.
# - `per_fold_metrics.csv` — every fold, every model, both supports.

# %%
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from analysis.loaders import CM_PER_DEGREE, CONTEXT_END_CM, CONTEXT_START_CM  # noqa: E402
from analysis.metrics import binned_error, error_summary  # noqa: E402
from analysis.results import (  # noqa: E402
    discover_studies,
    fold_predictions,
    load_final_results,
    pool_folds,
)

# %%
# --- config -----------------------------------------------------------------

BASELINE_DIR = Path(os.environ.get("BASELINE_DIR", ROOT / "results"))
VARIANT_DIR = Path(os.environ.get("VARIANT_DIR", ROOT / "results_posend515"))
PLOT_DIR = Path(os.environ.get("PLOT_DIR", VARIANT_DIR / "plots" / "comparison"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "pos"
BIN_CM = 2.0
MIN_COUNT = 30

MODEL_LABEL = {
    "baseline_full": "baseline (full support — reference only)",
    "baseline": f"baseline, true ≤ {CONTEXT_END_CM} cm",
    "variant": f"position-tail variant, true ≤ {CONTEXT_END_CM} cm",
}
MODEL_COLOR = {"baseline_full": "#CFD3D8", "baseline": "#7B848F", "variant": "#2A9D8F"}
METRICS = [("rmse", "RMSE (cm)"), ("mae", "MAE (cm)"), ("r2", "R²")]
INK = "#333333"
GRID = dict(color="0.85", linewidth=0.6)


def restricted(y_true_cm):
    return y_true_cm <= CONTEXT_END_CM


# %%
# --- load both trees, keep the datasets present in both -----------------------

def load_tree(results_dir):
    out = {}
    for info in discover_studies(results_dir, TARGET):
        final = load_final_results(results_dir, info.study)
        folds = [(t * CM_PER_DEGREE, p * CM_PER_DEGREE)
                 for t, p in fold_predictions(final, "position")]
        out[info.display] = {"info": info, "folds": folds}
    return out


print("=" * 100)
print(f"BASELINE vs POSITION-TAIL VARIANT — equal support (true ≤ {CONTEXT_END_CM} cm)")
print("=" * 100)
baseline = load_tree(BASELINE_DIR)
variant = load_tree(VARIANT_DIR)
common = [d for d in baseline if d in variant]
print(f"baseline: {len(baseline)} studies under {BASELINE_DIR}")
print(f"variant : {len(variant)} studies under {VARIANT_DIR}")
print(f"datasets in both: {common}")
if not common:
    raise SystemExit("Nothing to compare — has the variant batch been synced?")


# %%
# --- per-fold metrics for the three (model, support) combinations -----------

def fold_metrics(name):
    """Rows: one per (fold, model). Both models restricted to true ≤ CONTEXT_END_CM."""
    rows = []
    b_folds, v_folds = baseline[name]["folds"], variant[name]["folds"]
    if len(b_folds) != len(v_folds):
        raise ValueError(f"{name}: fold count differs ({len(b_folds)} vs {len(v_folds)})")
    for k, ((bt, bp), (vt, vp)) in enumerate(zip(b_folds, v_folds, strict=True), start=1):
        rows.append({"Dataset": name, "fold": k, "model": "baseline_full",
                     **error_summary(bt, bp)})
        rows.append({"Dataset": name, "fold": k, "model": "baseline",
                     **error_summary(bt, bp, restricted(bt))})
        rows.append({"Dataset": name, "fold": k, "model": "variant",
                     **error_summary(vt, vp, restricted(vt))})
    return rows


per_fold = pd.DataFrame([r for name in common for r in fold_metrics(name)])
pf_path = PLOT_DIR / "per_fold_metrics.csv"
per_fold.to_csv(pf_path, index=False, float_format="%.4f")
print(f"📁 {pf_path}")


# %%
# --- comparison table: medians + paired deltas --------------------------------

def paired_deltas(name):
    """variant − baseline(≤ctx) per fold, for each metric; plus how many folds improved."""
    sub = per_fold[per_fold.Dataset == name].pivot(index="fold", columns="model")
    out = {}
    for m, _ in METRICS:
        delta = sub[(m, "variant")] - sub[(m, "baseline")]
        better = (delta < 0) if m != "r2" else (delta > 0)
        out[f"delta_{m}"] = float(delta.median())
        out[f"folds_better_{m}"] = int(better.sum())
    return out


def dataset_row(name):
    sub = per_fold[per_fold.Dataset == name]
    med = sub.groupby("model")[["n", "rmse", "mae", "r2", "chance_rmse"]].median()
    row = {"Dataset": name, "Type": baseline[name]["info"].dtype, "n_folds": sub.fold.nunique()}
    for model in ("baseline_full", "baseline", "variant"):
        for m in ("n", "rmse", "mae", "r2", "chance_rmse"):
            row[f"{m}_{model}"] = med.loc[model, m]
    row.update(paired_deltas(name))
    return row


table = pd.DataFrame([dataset_row(n) for n in common])
csv_path = PLOT_DIR / "comparison_table.csv"
table.to_csv(csv_path, index=False, float_format="%.4f")
print(f"📁 {csv_path}")


def fmt(x, nd=1):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


def delta_cell(r, metric, nd):
    """`+0.3 (4/5)` — median paired delta and how many folds the variant won."""
    won, n = int(r[f"folds_better_{metric}"]), int(r["n_folds"])
    return f"{r[f'delta_{metric}']:+.{nd}f} ({won}/{n})"


def write_md(t, path):
    lines = [
        "# Baseline vs. position-tail variant — equal support",
        "",
        f"Generated by `exploration/09-comparison-position-tail.py`. Both models evaluated on "
        f"test bins with **true position ≤ {CONTEXT_END_CM} cm**; per-fold medians "
        "(5 paired folds). "
        "Δ = variant − baseline(≤ ctx); *folds better* counts folds where the variant wins.",
        "",
        "| Dataset | n bins/fold (median) base≤ctx / variant | RMSE base≤ctx | RMSE variant "
        "| Δ RMSE (folds better) | MAE base≤ctx | MAE variant | Δ MAE (folds better) "
        "| R² base≤ctx | R² variant | Δ R² (folds better) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['Dataset']} | {int(r['n_baseline']):,} / {int(r['n_variant']):,} "
            f"| {fmt(r['rmse_baseline'])} | {fmt(r['rmse_variant'])} "
            f"| {delta_cell(r, 'rmse', 1)} "
            f"| {fmt(r['mae_baseline'])} | {fmt(r['mae_variant'])} "
            f"| {delta_cell(r, 'mae', 1)} "
            f"| {fmt(r['r2_baseline'], 3)} | {fmt(r['r2_variant'], 3)} "
            f"| {delta_cell(r, 'r2', 3)} |")
    lines += [
        "",
        "## Reference: baseline on its full support (not comparable to the rows above)",
        "",
        "| Dataset | n bins | RMSE | MAE | R² | chance RMSE full / ≤ ctx |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['Dataset']} | {int(r['n_baseline_full']):,} | {fmt(r['rmse_baseline_full'])} "
            f"| {fmt(r['mae_baseline_full'])} | {fmt(r['r2_baseline_full'], 3)} "
            f"| {fmt(r['chance_rmse_baseline_full'])} / {fmt(r['chance_rmse_baseline'])} |")
    Path(path).write_text("\n".join(lines) + "\n")


md_path = PLOT_DIR / "comparison_table.md"
write_md(table, md_path)
print(f"📁 {md_path}")


# %%
# --- figure 1: paired per-fold boxplots, one panel per metric -----------------

def style_axes(ax):
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9, colors=INK)


def paired_boxplot(save_path):
    order = ["baseline_full", "baseline", "variant"]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.6 * len(METRICS), 5.2), dpi=200)
    for ax, (m, label) in zip(axes, METRICS, strict=True):
        sns.boxplot(data=per_fold, x="Dataset", y=m, hue="model", order=common,
                    hue_order=order, palette=MODEL_COLOR, ax=ax, showfliers=False,
                    linewidth=1.0, width=0.75)
        sns.stripplot(data=per_fold, x="Dataset", y=m, hue="model", order=common,
                      hue_order=order, dodge=True, color="0.25", size=3, alpha=0.6,
                      ax=ax, legend=False)
        ax.set_ylabel(label, fontsize=10, color=INK)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        style_axes(ax)
        if ax is not axes[0]:
            ax.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    fig.legend(handles, [MODEL_LABEL[k] for k in labels], loc="upper center", ncol=3,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"Position decoding, per outer fold — evaluated on true ≤ {CONTEXT_END_CM} cm "
                 "(grey box = baseline on its full support, reference only)",
                 fontsize=10.5, color=INK, y=0.93)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"📁 {save_path}")


paired_boxplot(PLOT_DIR / "metrics_paired_boxplot.svg")


# %%
# --- figure 2: error-by-position overlay (baseline vs variant) ----------------

def pooled_curve(folds):
    y_true, y_pred, _ = pool_folds(folds)
    mask = restricted(y_true)
    edges = np.arange(0, CONTEXT_END_CM + 2 * BIN_CM, BIN_CM)
    return binned_error(y_true[mask], y_pred[mask], edges, min_count=MIN_COUNT)


def shade_context(ax):
    ax.axvspan(CONTEXT_START_CM, CONTEXT_END_CM, color="gold", alpha=0.15, lw=0)


def overlay_column(axes, name):
    curves = {"baseline": pooled_curve(baseline[name]["folds"]),
              "variant": pooled_curve(variant[name]["folds"])}
    for model, tbl in curves.items():
        c = MODEL_COLOR[model]
        axes[0].plot(tbl["center"], tbl["mae"], color=c, linewidth=1.8, label=MODEL_LABEL[model])
        axes[0].plot(tbl["center"], tbl["rmse"], color=c, linewidth=1.2,
                     linestyle=(0, (4, 2)), label=f"RMSE — {MODEL_LABEL[model]}")
        axes[1].fill_between(tbl["center"], tbl["q25"], tbl["q75"], color=c, alpha=0.18, lw=0)
        axes[1].plot(tbl["center"], tbl["median_signed"], color=c, linewidth=1.8)
    axes[1].axhline(0, color="0.25", linewidth=0.9)
    for ax in axes:
        shade_context(ax)
        ax.set_xlim(0, CONTEXT_END_CM + BIN_CM)
        ax.grid(True, **GRID)
        style_axes(ax)
    axes[0].set_title(name, fontsize=10.5, color=INK, loc="left")
    axes[0].set_ylabel("MAE (solid) / RMSE (dashed), cm", fontsize=10, color=INK)
    axes[1].set_ylabel("Signed error (cm), median ± IQR", fontsize=10, color=INK)
    axes[1].set_xlabel("True position (cm)", fontsize=10, color=INK)


def overlay_figure(save_path):
    fig, axes = plt.subplots(2, len(common), figsize=(5.4 * len(common), 7.2), dpi=200,
                             squeeze=False, sharex=True)
    for col, name in enumerate(common):
        overlay_column(axes[:, col], name)
        if col > 0:
            for ax in axes[:, col]:
                ax.set_ylabel("")
    axes[0, 0].legend(loc="upper right", fontsize=7, frameon=False)
    fig.suptitle("Error along the corridor — baseline vs. position-tail variant "
                 f"(pooled folds, true ≤ {CONTEXT_END_CM} cm)", fontsize=11, color=INK,
                 x=0.02, ha="left", y=0.995)
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"📁 {save_path}")


overlay_figure(PLOT_DIR / "error_by_position_overlay.svg")


# %%
# --- compact table: bins / R² / RMSE / MSE, baseline vs variant --------------------

REGION_ORDER = {"Pir": 0, "CA3": 1, "DG": 2}


def dataset_order(name):
    return (REGION_ORDER.get(name.split("-")[0], 9), name)


def write_compact_md(path):
    bins = per_fold.groupby(["Dataset", "model"])["n"].sum().unstack()
    med = per_fold.groupby(["Dataset", "model"])[["r2", "rmse"]].median().unstack()
    lines = [
        "# Baseline vs. position-tail variant — compact",
        "",
        f"Both models evaluated on the same test bins (true position ≤ {CONTEXT_END_CM} cm; "
        "baseline restricted post hoc). Per-fold medians over 5 paired folds; bins = test "
        "bins pooled over the folds; MSE in cm².",
        "",
        "| Dataset | bins baseline | bins variant | R² baseline | R² variant "
        "| RMSE baseline | RMSE variant | MSE baseline | MSE variant |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(common, key=dataset_order):
        rb, rv = med.loc[name, ("rmse", "baseline")], med.loc[name, ("rmse", "variant")]
        lines.append(
            f"| {name} | {int(bins.loc[name, 'baseline']):,} | {int(bins.loc[name, 'variant']):,} "
            f"| {med.loc[name, ('r2', 'baseline')]:.2f} | {med.loc[name, ('r2', 'variant')]:.2f} "
            f"| {rb:.1f} | {rv:.1f} | {rb ** 2:.0f} | {rv ** 2:.0f} |")
    Path(path).write_text("\n".join(lines) + "\n")


compact_path = PLOT_DIR / "compact_table.md"
write_compact_md(compact_path)
print(f"📁 {compact_path}")


# %%
# --- per-zone comparison: where along the corridor does the variant gain? ------

ZONES_CM = {
    "0-10": (0.0, 10.0),
    "10-30": (10.0, 30.0),
    f"30-{CONTEXT_START_CM:g}": (30.0, CONTEXT_START_CM),
    f"ctx {CONTEXT_START_CM:g}-55": (CONTEXT_START_CM, 55.0),
    f"ctx 55-{CONTEXT_END_CM:g}": (55.0, CONTEXT_END_CM + 1e-6),
}


def zone_rows(name):
    rows = []
    for model, tree in (("baseline", baseline), ("variant", variant)):
        y_true, y_pred, _ = pool_folds(tree[name]["folds"])
        for zone, (lo, hi) in ZONES_CM.items():
            mask = (y_true >= lo) & (y_true < hi)
            rows.append({"Dataset": name, "model": model, "zone": zone,
                         **error_summary(y_true, y_pred, mask)})
    return rows


zones_long = pd.DataFrame([r for name in common for r in zone_rows(name)])
zone_tbl = zones_long.pivot(index=["Dataset", "zone"], columns="model",
                            values=["n", "bias", "mae"])
zone_tbl.columns = [f"{m}_{model}" for m, model in zone_tbl.columns]
zone_tbl["delta_mae"] = zone_tbl["mae_variant"] - zone_tbl["mae_baseline"]
zone_tbl = zone_tbl.reset_index()
zone_tbl["zone"] = pd.Categorical(zone_tbl["zone"], categories=list(ZONES_CM), ordered=True)
zone_tbl = zone_tbl.sort_values(["Dataset", "zone"]).reset_index(drop=True)
zc_path = PLOT_DIR / "zone_comparison.csv"
zone_tbl.to_csv(zc_path, index=False, float_format="%.4f")
print(f"📁 {zc_path}")


def write_zone_md(t, path):
    lines = [
        "# Where along the corridor does the variant gain? (pooled folds, cm)",
        "",
        "Δ MAE = variant − baseline on the bins of that zone (negative = variant better). "
        "Bias = mean(pred − true). The two models share the bins of every zone except the "
        "last one, where the variant's padding removed the bins closest to the context exit.",
        "",
        "| Dataset | zone (true cm) | n base / var | bias base | bias var "
        "| MAE base | MAE var | Δ MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in t.iterrows():
        lines.append(
            f"| {r['Dataset']} | {r['zone']} | {int(r['n_baseline']):,} / {int(r['n_variant']):,} "
            f"| {r['bias_baseline']:+.1f} | {r['bias_variant']:+.1f} "
            f"| {r['mae_baseline']:.1f} | {r['mae_variant']:.1f} | {r['delta_mae']:+.1f} |")
    Path(path).write_text("\n".join(lines) + "\n")


zmd_path = PLOT_DIR / "zone_comparison.md"
write_zone_md(zone_tbl, zmd_path)
print(f"📁 {zmd_path}")


# %%
# --- headline -----------------------------------------------------------------

print()
print("=" * 100)
print(f"HEADLINE — variant − baseline, both on true ≤ {CONTEXT_END_CM} cm (per-fold medians)")
print("=" * 100)
for _, r in table.iterrows():
    print(f"  {r['Dataset']:<16} ΔRMSE {delta_cell(r, 'rmse', 1):>12} cm   "
          f"ΔMAE {delta_cell(r, 'mae', 1):>12} cm   ΔR² {delta_cell(r, 'r2', 3)}   "
          f"[baseline≤ctx: RMSE {r['rmse_baseline']:.1f}, MAE {r['mae_baseline']:.1f}, "
          f"R² {r['r2_baseline']:.3f}]")
print()
print(f"  median ΔRMSE {table['delta_rmse'].median():+.1f} cm · "
      f"median ΔMAE {table['delta_mae'].median():+.1f} cm · "
      f"median ΔR² {table['delta_r2'].median():+.3f}")
print()
print("  Δ MAE by zone (median across datasets):")
for zone, sub in zone_tbl.groupby("zone", observed=True, sort=True):
    bias_b, bias_v = sub["bias_baseline"].median(), sub["bias_variant"].median()
    print(f"    {zone:<14} {sub['delta_mae'].median():+5.1f} cm   "
          f"(bias base {bias_b:+.1f} → var {bias_v:+.1f})")
