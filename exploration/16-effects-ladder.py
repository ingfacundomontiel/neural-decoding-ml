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
# # The effects ladder — one paired contrast per pipeline modification
#
# Every modification of the preprocessing pipeline explored in Phase 5 is
# measured as **one paired contrast** between two runs that differ in that
# factor only, on the **common bins** of the pair (kept sets reconstructed from
# the raw recording with the library's own filters, exact-verified against the
# pickles; the tail filter's padding follows each run's `bins_after`) and with
# **paired folds** (identical trial partition, asserted).
#
# | contrast | target | first − second | datasets |
# |---|---|---|---|
# | L1 tail removal (window kept, raw) | pos | posend515 − baseline | 5 pilots |
# | L2 window removal (no tail, raw) | pos | raw_w010 − posend515 | 5 pilots |
# | L3 smoothing (no tail, no window) | pos | smooth_w010 − raw_w010 | 17, identical rows |
# | L4 whole pipeline | pos | smooth_w010 − baseline | 17 |
# | L3-tail smoothing with the tail kept (2×2, ref.) | pos | tail_smooth − tail_raw | 6 piriform |
# | V3 smoothing (no tail, no window) | vel | smooth_w010 − raw_w010 | 17, identical rows |
# | V4 whole pipeline | vel | smooth_w010 − baseline | 17 |
#
# Outputs under `results/plots/effects-ladder/`: `effects_ladder.md` (per
# contrast: per-dataset deltas with folds won, aggregates for all / experts /
# naive, and the tail × smoothing interaction), `per_fold_metrics.csv`,
# `effects_ladder.svg`.

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
    get_idx_by_position_tail,
    periods_immobility,
)

from analysis.loaders import CM_PER_DEGREE, load_mat  # noqa: E402
from analysis.metrics import error_summary  # noqa: E402
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

TREES = {
    "baseline": {"results": "results", "data": "data/processed",
                 "window": (2, 2), "tail": None},
    "posend515": {"results": "results_posend515", "data": "data/processed_posend515",
                  "window": (2, 2), "tail": 515},
    "raw_w010": {"results": "results_notail_raw_w010",
                 "data": "data/processed_notail_raw_w010", "window": (0, 0), "tail": 515},
    "smooth_w010": {"results": "results_notail_smooth_w010",
                    "data": "data/processed_notail_smooth_w010", "window": (0, 0), "tail": 515},
    "tail_raw_w010": {"results": "results_raw_w010", "data": "data/processed_raw_w010",
                      "window": (0, 0), "tail": None},
    "tail_smooth_w010": {"results": "results_smooth200_w010",
                         "data": "data/processed_smooth200_w010", "window": (0, 0), "tail": None},
}
CONTRASTS = [
    ("L1 tail removal (window kept, raw)", "pos", "posend515", "baseline"),
    ("L2 window removal (no tail, raw)", "pos", "raw_w010", "posend515"),
    ("L3 smoothing (no tail, no window)", "pos", "smooth_w010", "raw_w010"),
    ("L4 whole pipeline", "pos", "smooth_w010", "baseline"),
    ("L3-tail smoothing (tail kept, no window)", "pos", "tail_smooth_w010", "tail_raw_w010"),
    ("V3 smoothing (no tail, no window)", "vel", "smooth_w010", "raw_w010"),
    ("V4 whole pipeline", "vel", "smooth_w010", "baseline"),
]
PICKLE_VARIANT = {"pos": "_withCtxt_onlyPosition_flat.pickle",
                  "vel": "_withCtxt_onlyVelocity_flat.pickle"}
UNIT = {"pos": "cm", "vel": "cm/s"}
PLOT_DIR = Path(os.environ.get("PLOT_DIR", ROOT / "results" / "plots" / "effects-ladder"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REGION_ORDER = {"Pir": 0, "CA3": 1, "DG": 2}
INK = "#333333"
COHORT_COLOR = {"expert": "#004E89", "naive": "#E76F51"}


# %%
# --- kept-set reconstruction + run loading ---------------------------------------

def raw_context(info):
    mat = load_mat(mat_path(ROOT / "data" / "raw", info))
    return {"pos": np.asarray(mat["position"], float).ravel(),
            "vel": np.asarray(mat["velocity"], float).ravel(),
            "tfb": np.asarray(mat["trialFinalBin"], int).ravel() - 1,
            "dprime": np.asarray(mat["dPrime"], float).ravel(),
            "thresh": 2 if info.cohort == "expert" else -1.1, "_kept": {}}


def kept_raw_indices(ctx, bins_before, bins_after, tail):
    key = (bins_before, bins_after, tail)
    if key in ctx["_kept"]:
        return ctx["_kept"][key]
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
    if tail is not None:
        removed = np.union1d(removed, get_idx_by_position_tail(
            ctx["pos"], ctx["tfb"], tail, bins_after))
    ctx["_kept"][key] = np.setdiff1d(np.arange(n_raw), removed.astype(int))
    return ctx["_kept"][key]


def load_run(info, ctx, tree, target):
    cfg = TREES[tree]
    study = f"mlp_{info.stem}_{target}"
    if not (ROOT / cfg["results"] / study / "final_results.json").is_file():
        return None
    with pickle_path(ROOT / cfg["data"], info, PICKLE_VARIANT[target]).open("rb") as f:
        _, y, trial_markers = pickle.load(f)  # noqa: S301 — our own artefact
    kept = kept_raw_indices(ctx, *cfg["window"], cfg["tail"])
    if len(kept) != len(y) or not np.allclose(ctx[target][kept], y[:, 0]):
        raise ValueError(f"{info.display}/{tree}/{target}: kept set != pickle")
    rows = pooled_test_rows(ROOT / cfg["results"], study, y[:, 0], trial_markers)
    scale = CM_PER_DEGREE if target == "pos" else 1.0
    y_true, y_pred, fold_id = pool_folds(
        [(t * scale, p * scale)
         for t, p in fold_predictions(load_final_results(ROOT / cfg["results"], study))])
    return {"y_true": y_true, "y_pred": y_pred, "fold_id": fold_id, "raw_idx": kept[rows],
            "trials": fold_test_trials(ROOT / cfg["results"], study)}


def paired_fold_metrics(run_a, run_b):
    """Per-fold metrics of both runs on their common bins; folds must be paired."""
    assert run_a["trials"] == run_b["trials"], "fold partition differs"
    common = np.intersect1d(run_a["raw_idx"], run_b["raw_idx"])
    out = {}
    for tag, d in (("a", run_a), ("b", run_b)):
        m_common = np.isin(d["raw_idx"], common)
        out[tag] = pd.DataFrame([
            {"fold": int(k), **error_summary(d["y_true"], d["y_pred"],
                                             (d["fold_id"] == k) & m_common)}
            for k in np.unique(d["fold_id"])]).set_index("fold")
    return out, len(common)


# %%
# --- run every contrast -----------------------------------------------------------

print("=" * 100)
print("EFFECTS LADDER — paired contrasts on common bins")
print("=" * 100)
infos = {i.display: i for i in discover_studies(ROOT / "results", "pos")}
names = sorted(infos, key=lambda n: (REGION_ORDER.get(n.split("-")[0], 9), n))
cache: dict = {}
per_fold_rows, delta_rows = [], []
for label, target, tree_a, tree_b in CONTRASTS:
    for name in names:
        info = infos[name]
        ctx = cache.setdefault(("ctx", name), None) or raw_context(info)
        cache[("ctx", name)] = ctx
        runs = {}
        for tree in (tree_a, tree_b):
            key = (name, tree, target)
            if key not in cache:
                cache[key] = load_run(info, ctx, tree, target)
            runs[tree] = cache[key]
        if runs[tree_a] is None or runs[tree_b] is None:
            continue
        m, n_common = paired_fold_metrics(runs[tree_a], runs[tree_b])
        for tag, tree in (("a", tree_a), ("b", tree_b)):
            for k, r in m[tag].iterrows():
                per_fold_rows.append({"contrast": label, "target": target, "Dataset": name,
                                      "run": tree, "fold": k, **r.to_dict()})
        row = {"contrast": label, "target": target, "Dataset": name,
               "cohort": info.cohort, "n_common": n_common}
        for metric in ("r2", "rmse", "mae"):
            delta = m["a"][metric] - m["b"][metric]
            better = (delta > 0) if metric == "r2" else (delta < 0)
            row[f"d_{metric}"] = float(delta.median())
            row[f"won_{metric}"] = int(better.sum())
            row[f"{metric}_a"] = float(m["a"][metric].median())
            row[f"{metric}_b"] = float(m["b"][metric].median())
        delta_rows.append(row)
    n_ds = sum(1 for r in delta_rows if r["contrast"] == label)
    print(f"  {label:<44} {n_ds:>2} datasets")

per_fold = pd.DataFrame(per_fold_rows)
per_fold.to_csv(PLOT_DIR / "per_fold_metrics.csv", index=False, float_format="%.4f")
deltas = pd.DataFrame(delta_rows)
deltas.to_csv(PLOT_DIR / "deltas.csv", index=False, float_format="%.4f")


# %%
# --- markdown -----------------------------------------------------------------------

def agg_cell(sub, metric, nd):
    if sub.empty:
        return "—"
    return (f"{sub[f'd_{metric}'].median():+.{nd}f} "
            f"({int(sub[f'won_{metric}'].sum())}/{5 * len(sub)})")


def contrast_block(label, target, tree_a, tree_b):
    sub = deltas[deltas.contrast == label]
    u = UNIT[target]
    lines = [f"## {label}", "",
             f"`{tree_a}` − `{tree_b}`, target **{target}**, {len(sub)} datasets. "
             f"Per-fold medians; Δ = first − second (folds won by the first).", "",
             f"| Dataset | cohort | n common | R² {tree_b} | R² {tree_a} | Δ R² | "
             f"RMSE {tree_b} | RMSE {tree_a} | Δ RMSE ({u}) | MAE {tree_b} | MAE {tree_a} "
             f"| Δ MAE ({u}) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in sub.iterrows():
        lines.append(
            f"| {r['Dataset']} | {r['cohort']} | {int(r['n_common']):,} "
            f"| {r['r2_b']:.3f} | {r['r2_a']:.3f} | {r['d_r2']:+.3f} ({int(r['won_r2'])}/5) "
            f"| {r['rmse_b']:.1f} | {r['rmse_a']:.1f} "
            f"| {r['d_rmse']:+.1f} ({int(r['won_rmse'])}/5) "
            f"| {r['mae_b']:.1f} | {r['mae_a']:.1f} | {r['d_mae']:+.1f} ({int(r['won_mae'])}/5) |")
    lines += ["", "| aggregate | Δ R² | Δ RMSE | Δ MAE |", "|---|---:|---:|---:|"]
    for tag, s in (("all", sub), ("experts", sub[sub.cohort == "expert"]),
                   ("naive", sub[sub.cohort == "naive"])):
        lines.append(f"| {tag} ({len(s)}) | {agg_cell(s, 'r2', 3)} | {agg_cell(s, 'rmse', 1)} "
                     f"| {agg_cell(s, 'mae', 1)} |")
    return lines + [""]


def interaction_block():
    a = deltas[deltas.contrast.str.startswith("L3 smoothing")].set_index("Dataset")
    b = deltas[deltas.contrast.str.startswith("L3-tail")].set_index("Dataset")
    both = a.index.intersection(b.index)
    lines = ["## Interaction — smoothing effect with vs without the tail (piriform)", "",
             "Same factor (smoothing, no window) measured in the two tail worlds. "
             "Additivity would give equal deltas.", "",
             "| Dataset | Δ R² smoothing, tail kept | Δ R² smoothing, no tail | difference |",
             "|---|---:|---:|---:|"]
    for ds in both:
        lines.append(f"| {ds} | {b.loc[ds, 'd_r2']:+.3f} | {a.loc[ds, 'd_r2']:+.3f} "
                     f"| {a.loc[ds, 'd_r2'] - b.loc[ds, 'd_r2']:+.3f} |")
    if len(both):
        lines.append(f"| **median** | {b.loc[both, 'd_r2'].median():+.3f} "
                     f"| {a.loc[both, 'd_r2'].median():+.3f} "
                     f"| {(a.loc[both, 'd_r2'] - b.loc[both, 'd_r2']).median():+.3f} |")
    return lines + [""]


md = ["# Effects ladder — Phase 5 pipeline modifications", "",
      "Generated by `exploration/16-effects-ladder.py`. Each contrast pairs two runs that "
      "differ in one factor, evaluated on the common bins of the pair (kept-set intersection, "
      "exact-verified) with paired folds. R² is comparable within a contrast only.", ""]
for label, target, tree_a, tree_b in CONTRASTS:
    md += contrast_block(label, target, tree_a, tree_b)
md += interaction_block()
(PLOT_DIR / "effects_ladder.md").write_text("\n".join(md) + "\n")
print(f"📁 {(PLOT_DIR / 'effects_ladder.md').relative_to(ROOT)}")


# %%
# --- figure: Δ per contrast, dots per dataset, median bar ------------------------------

def ladder_figure(save_path):
    labels = [c[0] for c in CONTRASTS]
    short = ["L1 −tail", "L2 −window", "L3 +smooth", "L4 pipeline", "L3 +smooth (tail)",
             "V3 +smooth", "V4 pipeline"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), dpi=200, sharex=True)
    rng = np.random.default_rng(0)
    for ax, metric, ylabel in ((axes[0], "d_r2", "Δ R²"), (axes[1], "d_mae", "Δ MAE (cm · cm/s)")):
        for i, label in enumerate(labels):
            sub = deltas[deltas.contrast == label]
            if sub.empty:
                continue
            ax.bar(i, sub[metric].median(), width=0.55, color="0.85", zorder=1)
            for _, r in sub.iterrows():
                ax.scatter(i + rng.uniform(-0.18, 0.18), r[metric], s=22, zorder=3,
                           color=COHORT_COLOR[r["cohort"]], alpha=0.85, edgecolors="white",
                           linewidths=0.4)
        ax.axhline(0, color="0.25", linewidth=0.9)
        ax.axvline(4.5, color="0.6", linewidth=0.8, linestyle=":")
        ax.set_ylabel(ylabel, fontsize=10, color=INK)
        ax.grid(axis="y", color="0.85", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9, colors=INK)
    axes[1].set_xticks(range(len(short)), short, fontsize=9, rotation=15)
    for cohort, color in COHORT_COLOR.items():
        axes[0].scatter([], [], color=color, s=22, label=cohort)
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")
    axes[0].set_title("Effects ladder — paired contrasts on common bins "
                      "(grey bar = median across datasets; position | velocity)",
                      fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(save_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"📁 {save_path.relative_to(ROOT)}")


ladder_figure(PLOT_DIR / "effects_ladder.svg")


# %%
# --- headline --------------------------------------------------------------------------

print()
print("=" * 100)
print("HEADLINE — median Δ across datasets (folds won / total)")
print("=" * 100)
for label, _target, _, _ in CONTRASTS:
    sub = deltas[deltas.contrast == label]
    exp, nai = sub[sub.cohort == "expert"], sub[sub.cohort == "naive"]
    print(f"  {label:<44} ΔR² {agg_cell(sub, 'r2', 3):>14}  "
          f"ΔRMSE {agg_cell(sub, 'rmse', 1):>13}  ΔMAE {agg_cell(sub, 'mae', 1):>13}  "
          f"[experts {agg_cell(exp, 'r2', 3)} · naive {agg_cell(nai, 'r2', 3)}]")
