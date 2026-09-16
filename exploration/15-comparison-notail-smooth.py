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
# # Batch 007 — no-tail + smoothing + no window, vs the other pipelines
#
# Three runs per dataset:
#
# | run | tail filter | activity | window |
# |---|---|---|---|
# | baseline (Batch 002) | off | raw | 2/1/2 |
# | posend515 (Batches 003/004, 5 pilots) | 515° | raw | 2/1/2 |
# | **new (Batch 007, 17 datasets)** | 515° | smoothed σ=200 ms | 0/1/0 |
#
# Every comparison is on **common bins** (the intersection of the runs' kept
# sets, reconstructed from the raw recording with the library's own filters and
# exact-verified against the pickles; the tail filter's padding follows each
# run's `bins_after`). Folds are paired (same trial partition — asserted).
#
# - **Table A** (5 pilot datasets): three-way, on the three runs' common bins.
# - **Table B** (all 17): new vs baseline, on their common bins.
#
# Output: `results_notail_smooth_w010/plots/comparison/notail_smooth_vs_others.md`
# (+ per-fold CSV).

# %%
import os
import pickle
import sys
from pathlib import Path

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

RUNS = {
    "baseline": {"results": "results", "data": "data/processed",
                 "window": (2, 2), "tail": None},
    "posend515": {"results": "results_posend515", "data": "data/processed_posend515",
                  "window": (2, 2), "tail": 515},
    "new": {"results": "results_notail_smooth_w010",
            "data": "data/processed_notail_smooth_w010", "window": (0, 0), "tail": 515},
}
PLOT_DIR = Path(os.environ.get(
    "PLOT_DIR", ROOT / "results_notail_smooth_w010" / "plots" / "comparison"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)
REGION_ORDER = {"Pir": 0, "CA3": 1, "DG": 2}


# %%
# --- kept-set reconstruction (window- and tail-aware) --------------------------

def raw_context(info):
    mat = load_mat(mat_path(ROOT / "data" / "raw", info))
    return {"pos": np.asarray(mat["position"], float).ravel(),
            "vel": np.asarray(mat["velocity"], float).ravel(),
            "tfb": np.asarray(mat["trialFinalBin"], int).ravel() - 1,
            "dprime": np.asarray(mat["dPrime"], float).ravel(),
            "thresh": 2 if info.cohort == "expert" else -1.1}


def kept_raw_indices(ctx, bins_before, bins_after, tail):
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
    return np.setdiff1d(np.arange(n_raw), removed.astype(int))


def load_run(info, ctx, cfg):
    pkl = pickle_path(ROOT / cfg["data"], info, "_withCtxt_onlyPosition_flat.pickle")
    with pkl.open("rb") as f:
        _, y, trial_markers = pickle.load(f)  # noqa: S301 — our own artefact
    kept = kept_raw_indices(ctx, *cfg["window"], cfg["tail"])
    if len(kept) != len(y) or not np.allclose(ctx["pos"][kept], y[:, 0]):
        raise ValueError(f"{info.display} / {cfg['results']}: kept set != pickle")
    rows = pooled_test_rows(ROOT / cfg["results"], info.study, y[:, 0], trial_markers)
    final = load_final_results(ROOT / cfg["results"], info.study)
    y_true, y_pred, fold_id = pool_folds(
        [(t * CM_PER_DEGREE, p * CM_PER_DEGREE)
         for t, p in fold_predictions(final, "position")])
    return {"y_true": y_true, "y_pred": y_pred, "fold_id": fold_id,
            "raw_idx": kept[rows],
            "trials": fold_test_trials(ROOT / cfg["results"], info.study)}


def load_dataset(info):
    ctx = raw_context(info)
    runs = {}
    for name, cfg in RUNS.items():
        if (ROOT / cfg["results"] / info.study / "final_results.json").is_file():
            runs[name] = load_run(info, ctx, cfg)
    ref = runs["new"]["trials"]
    for name, d in runs.items():
        assert d["trials"] == ref, f"{info.display}/{name}: fold partition differs"
    common = None
    for d in runs.values():
        common = d["raw_idx"] if common is None else np.intersect1d(common, d["raw_idx"])
    for d in runs.values():
        d["common"] = np.isin(d["raw_idx"], common)
    return runs


# %%
# --- per-fold metrics on the common bins ---------------------------------------

def per_fold_rows(name, runs):
    rows = []
    for run_name, d in runs.items():
        for k in np.unique(d["fold_id"]):
            m = (d["fold_id"] == k) & d["common"]
            rows.append({"Dataset": name, "run": run_name, "fold": int(k),
                         **error_summary(d["y_true"], d["y_pred"], m)})
    return rows


print("=" * 100)
print("BATCH 007 (no tail + smooth + no window) vs posend515 and baseline — common bins")
print("=" * 100)
infos = {i.display: i for i in discover_studies(ROOT / "results_notail_smooth_w010", "pos")}
names = sorted(infos, key=lambda n: (REGION_ORDER.get(n.split("-")[0], 9), n))
all_rows = []
n_runs = {}
for name in names:
    runs = load_dataset(infos[name])
    n_runs[name] = list(runs)
    all_rows += per_fold_rows(name, runs)
    print(f"  ✅ {name}: runs {list(runs)} · common bins "
          f"{int(runs['new']['common'].sum()):,}")
per_fold = pd.DataFrame(all_rows)
per_fold.to_csv(PLOT_DIR / "notail_smooth_vs_others_per_fold.csv",
                index=False, float_format="%.4f")
med = per_fold.groupby(["Dataset", "run"], sort=False)[["n", "r2", "rmse", "mae"]].median()


def paired(ds, run_a, run_b, metric):
    sub = per_fold[per_fold.Dataset == ds]
    a = sub[sub.run == run_a].set_index("fold")[metric]
    b = sub[sub.run == run_b].set_index("fold")[metric]
    delta = a - b
    better = (delta > 0) if metric == "r2" else (delta < 0)
    return float(delta.median()), int(better.sum()), len(delta)


def cell(ds, a, b, metric, nd):
    d, w, n = paired(ds, a, b, metric)
    return f"{d:+.{nd}f} ({w}/{n})"


# %%
# --- markdown -------------------------------------------------------------------

def block(names_subset, runs_order, ref_pairs, title, note):
    lines = [f"## {title}", "", note, ""]
    hdr = "| Dataset | " + " | ".join(
        f"R² {r} | RMSE {r} | MAE {r}" for r in runs_order)
    for _a, _b, lab in ref_pairs:
        hdr += f" | Δ R² {lab} | Δ RMSE {lab} | Δ MAE {lab}"
    lines.append(hdr + " |")
    lines.append("|---|" + "---:|" * (3 * len(runs_order) + 3 * len(ref_pairs)))
    for ds in names_subset:
        row = f"| {ds} |"
        for r in runs_order:
            v = med.loc[(ds, r)]
            row += f" {v['r2']:.3f} | {v['rmse']:.1f} | {v['mae']:.1f} |"
        for a, b, _ in ref_pairs:
            row += (f" {cell(ds, a, b, 'r2', 3)} | {cell(ds, a, b, 'rmse', 1)} "
                    f"| {cell(ds, a, b, 'mae', 1)} |")
        lines.append(row)
    # aggregate row (median of per-dataset deltas, folds won summed)
    agg = "| **median (folds won summed)** |" + " — | — | — |" * len(runs_order)
    for a, b, _ in ref_pairs:
        for metric, nd in (("r2", 3), ("rmse", 1), ("mae", 1)):
            stats = [paired(ds, a, b, metric) for ds in names_subset]
            agg += (f" {np.median([d for d, _, _ in stats]):+.{nd}f} "
                    f"({sum(w for _, w, _ in stats)}/{sum(n for _, _, n in stats)}) |")
    lines.append(agg)
    return lines


pilots = [n for n in names if "posend515" in n_runs[n]]
lines = [
    "# Batch 007 (no tail + smoothing + no window) vs the other pipelines",
    "",
    "Generated by `exploration/15-comparison-notail-smooth.py`. Per-fold medians, 5 paired "
    "folds; MAE/RMSE in cm. Every block is evaluated on the **common bins** of its runs "
    "(kept-set intersection, exact-verified), so R² values are comparable within a block "
    "but not across blocks or with previously reported own-support numbers.",
    "",
]
lines += block(
    pilots, ["baseline", "posend515", "new"],
    [("new", "posend515", "new−posend515"), ("new", "baseline", "new−baseline")],
    "A — the 5 pilot datasets, three-way",
    "Common bins of the three runs (= the posend515 kept set, the strictest).")
lines += [""]
lines += block(
    names, ["baseline", "new"],
    [("new", "baseline", "new−baseline")],
    "B — all 17 datasets, new pipeline vs baseline",
    "Common bins of the two runs: baseline kept set ∩ no-tail (no padding) kept set — "
    "i.e. the corridor up to the context exit.")
out = PLOT_DIR / "notail_smooth_vs_others.md"
out.write_text("\n".join(lines) + "\n")
print(f"📁 {out.relative_to(ROOT)}")


# %%
# --- headline -------------------------------------------------------------------

print()
print("=" * 100)
print("HEADLINE")
print("=" * 100)
for metric, nd in (("r2", 3), ("rmse", 1), ("mae", 1)):
    s3 = [paired(ds, "new", "posend515", metric) for ds in pilots]
    s2 = [paired(ds, "new", "baseline", metric) for ds in names]
    print(f"  Δ {metric.upper():<4} new−posend515 (5 pilotos): "
          f"{np.median([d for d, _, _ in s3]):+.{nd}f} "
          f"({sum(w for _, w, _ in s3)}/{sum(n for _, _, n in s3)})   ·   "
          f"new−baseline (17): {np.median([d for d, _, _ in s2]):+.{nd}f} "
          f"({sum(w for _, w, _ in s2)}/{sum(n for _, _, n in s2)})")
