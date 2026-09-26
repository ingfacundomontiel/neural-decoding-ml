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
# # Velocity checks — where the velocity decoder fails, and pos↔vel error coupling
#
# Two descriptive checks on the two complete pipelines (baseline: raw + window
# + tail; new: smoothed + no window + no tail), each on its own test bins:
#
# 1. **Velocity error along the corridor.** MAE of the velocity decoder (cm/s)
#    per 2 cm bin of *true position* (position read from the pickle through
#    the reconstructed rows). Never looked at before: does the velocity error
#    concentrate in the tail like the position error, or elsewhere (starts,
#    pauses)?
# 2. **pos↔vel error coupling per pipeline** (decoupling analysis A3, extended
#    to all 17 datasets and both pipelines). Within-position (2 cm strata,
#    robust z) correlation between |position error| and |velocity error|, per
#    fold. If smoothing raises it, the smoothed input couples both decoders to
#    the same temporal state — a hidden cost of the new pipeline.
#
# Outputs under `results/plots/velocity-checks/`: one `<dataset>_velocity_error.svg`
# per dataset, `summary_table.md/csv`.

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

PIPELINES = {
    "baseline": {"results": "results", "data": "data/processed", "color": "#7B848F"},
    "new": {"results": "results_notail_smooth_w010",
            "data": "data/processed_notail_smooth_w010", "color": "#2A9D8F"},
}
PLOT_DIR = Path(os.environ.get("PLOT_DIR", ROOT / "results" / "plots" / "velocity-checks"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)
BIN_CM, MIN_COUNT = 2.0, 30
ZONES = {"pre": (-np.inf, CONTEXT_START_CM), "ctx": (CONTEXT_START_CM, CONTEXT_END_CM),
         "post": (CONTEXT_END_CM, np.inf)}
REGION_ORDER = {"Pir": 0, "CA3": 1, "DG": 2}
INK = "#333333"


# %%
# --- load: aligned pos + vel predictions with per-bin true position ------------

def load_pipeline(info, cfg):
    out = {}
    rows_ref = None
    for target, variant in (("pos", "_withCtxt_onlyPosition_flat.pickle"),
                            ("vel", "_withCtxt_onlyVelocity_flat.pickle")):
        study = f"mlp_{info.stem}_{target}"
        with pickle_path(ROOT / cfg["data"], info, variant).open("rb") as f:
            _, y, trial_markers = pickle.load(f)  # noqa: S301 — our own artefact
        rows = pooled_test_rows(ROOT / cfg["results"], study, y[:, 0], trial_markers)
        if rows_ref is not None and not np.array_equal(rows, rows_ref):
            raise ValueError(f"{info.display}/{cfg['results']}: pos and vel rows differ")
        rows_ref = rows
        scale = CM_PER_DEGREE if target == "pos" else 1.0
        y_true, y_pred, fold_id = pool_folds(
            [(t * scale, p * scale)
             for t, p in fold_predictions(load_final_results(ROOT / cfg["results"], study))])
        out[target] = (y_true, y_pred)
        out["fold_id"] = fold_id
    out["pos_true"] = out["pos"][0]
    out["pos_err"] = np.abs(out["pos"][1] - out["pos"][0])
    out["vel_err"] = np.abs(out["vel"][1] - out["vel"][0])
    return out


def robust_z(values, groups):
    z = np.full(len(values), np.nan)
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < MIN_COUNT:
            continue
        med = np.median(values[m])
        mad = 1.4826 * np.median(np.abs(values[m] - med))
        if mad > 0:
            z[m] = (values[m] - med) / mad
    return z


def error_coupling_per_fold(d):
    """Within-position correlation of |pos err| and |vel err|, one value per fold."""
    strata = np.digitize(d["pos_true"], np.arange(0, 120, BIN_CM)) - 1
    zp, zv = robust_z(d["pos_err"], strata), robust_z(d["vel_err"], strata)
    vals = []
    for k in np.unique(d["fold_id"]):
        m = (d["fold_id"] == k) & np.isfinite(zp) & np.isfinite(zv)
        vals.append(float(np.corrcoef(zp[m], zv[m])[0, 1]) if m.sum() > 30 else np.nan)
    return vals


def vel_mae_by_position(d, edges):
    idx = np.digitize(d["pos_true"], edges) - 1
    mae = np.full(len(edges) - 1, np.nan)
    for i in range(len(edges) - 1):
        m = idx == i
        if m.sum() >= MIN_COUNT:
            mae[i] = np.mean(d["vel_err"][m])
    return mae


# %%
# --- run --------------------------------------------------------------------------

print("=" * 100)
print("VELOCITY CHECKS — error along the corridor + pos↔vel error coupling, both pipelines")
print("=" * 100)
infos = {i.display: i for i in discover_studies(ROOT / "results", "vel")}
names = sorted(infos, key=lambda n: (REGION_ORDER.get(n.split("-")[0], 9), n))
rows = []
for name in names:
    info = infos[name]
    data = {p: load_pipeline(info, cfg) for p, cfg in PIPELINES.items()}
    hi = float(np.ceil(np.percentile(data["baseline"]["pos_true"], 99.9) / 10) * 10)
    edges = np.arange(0, hi + BIN_CM, BIN_CM)
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    row = {"Dataset": name, "cohort": info.cohort}
    for p, d in data.items():
        ax.plot(0.5 * (edges[:-1] + edges[1:]), vel_mae_by_position(d, edges),
                color=PIPELINES[p]["color"], linewidth=1.7, label=f"{p} (own test bins)")
        coupling = error_coupling_per_fold(d)
        row[f"coupling_{p}"] = float(np.nanmedian(coupling))
        for zone, (lo, hi_z) in ZONES.items():
            m = (d["pos_true"] >= lo) & (d["pos_true"] < hi_z)
            row[f"vel_mae_{zone}_{p}"] = (float(np.mean(d["vel_err"][m]))
                                          if m.sum() >= MIN_COUNT else np.nan)
        row[f"vel_mae_all_{p}"] = float(np.mean(d["vel_err"]))
    ax.axvspan(CONTEXT_START_CM, CONTEXT_END_CM, color="gold", alpha=0.15, lw=0)
    ax.axvline(CONTEXT_END_CM, color="0.35", linewidth=0.8, linestyle=":")
    ax.set_xlabel("True position (cm)", fontsize=10, color=INK)
    ax.set_ylabel("Velocity MAE (cm/s)", fontsize=10, color=INK)
    ax.set_title(f"{name} — velocity decoding error along the corridor",
                 fontsize=10.5, color=INK, loc="left")
    ax.grid(True, color="0.85", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{name}_velocity_error.svg", format="svg", bbox_inches="tight")
    plt.close(fig)
    rows.append(row)
    print(f"  ✅ {name}")

summary = pd.DataFrame(rows)
summary.to_csv(PLOT_DIR / "summary_table.csv", index=False, float_format="%.4f")


# %%
# --- markdown + headline ----------------------------------------------------------

def fmt(x, nd=1):
    return "—" if pd.isna(x) else f"{x:.{nd}f}"


lines = [
    "# Velocity checks — error along the corridor and pos↔vel error coupling",
    "",
    "Generated by `exploration/17-velocity-checks.py`. Each pipeline on its own test bins "
    "(baseline includes the tail; new has none). Velocity MAE in cm/s. **coupling** = "
    "within-position correlation of |pos error| and |vel error| (median over folds); higher "
    "means the two decoders fail on the same bins.",
    "",
    "| Dataset | cohort | vel MAE pre (base / new) | vel MAE ctx (base / new) "
    "| vel MAE post (base) | coupling base | coupling new | Δ coupling |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for _, r in summary.iterrows():
    lines.append(
        f"| {r['Dataset']} | {r['cohort']} "
        f"| {fmt(r['vel_mae_pre_baseline'])} / {fmt(r['vel_mae_pre_new'])} "
        f"| {fmt(r['vel_mae_ctx_baseline'])} / {fmt(r['vel_mae_ctx_new'])} "
        f"| {fmt(r['vel_mae_post_baseline'])} "
        f"| {r['coupling_baseline']:+.2f} | {r['coupling_new']:+.2f} "
        f"| {r['coupling_new'] - r['coupling_baseline']:+.2f} |")
d_coupling = summary["coupling_new"] - summary["coupling_baseline"]
lines += ["", "## Medians across datasets", "",
          "- vel MAE pre / ctx / post (baseline): "
          f"{summary['vel_mae_pre_baseline'].median():.1f} / "
          f"{summary['vel_mae_ctx_baseline'].median():.1f} / "
          f"{summary['vel_mae_post_baseline'].median():.1f} cm/s",
          f"- vel MAE pre / ctx (new): {summary['vel_mae_pre_new'].median():.1f} / "
          f"{summary['vel_mae_ctx_new'].median():.1f} cm/s",
          f"- coupling baseline → new: {summary['coupling_baseline'].median():+.2f} → "
          f"{summary['coupling_new'].median():+.2f} "
          f"(Δ median {(summary['coupling_new'] - summary['coupling_baseline']).median():+.2f}, "
          f"raised in {int((d_coupling > 0).sum())}/{len(summary)})"]
(PLOT_DIR / "summary_table.md").write_text("\n".join(lines) + "\n")
print(f"📁 {(PLOT_DIR / 'summary_table.md').relative_to(ROOT)}")
print()
print("\n".join(lines[-4:]))
