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
# # Distribution of decoded vs true positions — shrinkage check, test 1
#
# An MSE-trained decoder facing ambiguous input predicts close to the mean of its
# training targets. If that happens a lot, the decoded positions pile up around
# the mean and are missing at the ends of the corridor, compared with where the
# animal actually was. Counting predictions near the mean alone is not enough —
# the animal's own occupancy is uneven — so the test compares the two
# distributions on the same bins: the true positions and the decoded positions of
# the pooled test folds (one out-of-sample prediction per kept bin).
#
# Final pipeline (`results_notail_smooth_w010/`: tail 515°, Gaussian smoothing
# σ = 200 ms, window 0/1/0), position target. One figure per recording
# (`RECORDINGS`, default Pir_N31 expert and Pir_N53 naive). The dashed line is the
# mean true position of the kept bins (≈ each fold's training mean).
#
# Second figure per recording (test 1b): the decoded-position histogram split into
# **correct** predictions (within ±`TOLERANCE_CM` of the true position) and
# incorrect ones — of the predictions landing at each place, how many are right?
#
# Outputs under `results/plots/shrinkage/` (`PLOT_DIR` overrides).

# %%
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from analysis.loaders import CM_PER_DEGREE, CONTEXT_END_CM, CONTEXT_START_CM  # noqa: E402
from analysis.results import (  # noqa: E402
    discover_studies,
    fold_predictions,
    fold_r2_scores,
    load_final_results,
    pool_folds,
)

RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results_notail_smooth_w010"))
PLOT_DIR = Path(os.environ.get("PLOT_DIR", ROOT / "results" / "plots" / "shrinkage"))
PLOT_DIR.mkdir(parents=True, exist_ok=True)
RECORDINGS = os.environ.get("RECORDINGS", "Pir-Expert-N31,Pir-Naive-N53").split(",")
BIN_CM = 2.0
TOLERANCE_CM = float(os.environ.get("TOLERANCE_CM", "5"))
X_RANGE = (-10.0, 80.0)   # same frame for every recording; predictions outside are counted
COLOR = {"expert": "#0B63A5", "naive": "#7FB1DA"}
TRUE_COLOR = "#9A9A9A"

# %%
studies = {i.display: i for i in discover_studies(RESULTS_DIR, "pos")}


def pooled_cm(display: str):
    """Pooled test (true, decoded) positions in cm, plus the fold-median R²."""
    final = load_final_results(RESULTS_DIR, studies[display].study)
    y_true, y_pred, _ = pool_folds(
        [(t * CM_PER_DEGREE, p * CM_PER_DEGREE) for t, p in fold_predictions(final)])
    return y_true, y_pred, float(np.median(fold_r2_scores(final)))


def distribution_figure(display: str) -> Path:
    y_true, y_pred, r2 = pooled_cm(display)
    edges = np.arange(X_RANGE[0], X_RANGE[1] + BIN_CM, BIN_CM)
    outside = np.mean((y_pred < X_RANGE[0]) | (y_pred > X_RANGE[1])) * 100
    weights = np.full(len(y_true), 100.0 / len(y_true))  # % of test bins
    color = COLOR[studies[display].cohort]

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.axvspan(CONTEXT_START_CM, CONTEXT_END_CM, color="#FFF3C4", lw=0, zorder=0)
    ax.hist(y_true, bins=edges, weights=weights, color=TRUE_COLOR, alpha=0.55,
            label="true position")
    ax.hist(y_pred, bins=edges, weights=weights, histtype="step", color=color, lw=2.4,
            label="decoded position")
    ax.axvline(y_true.mean(), color="#1E1E1E", lw=1.4, ls=(0, (5, 3)),
               label=f"mean true position ({y_true.mean():.1f} cm)")
    ax.set_xlim(*X_RANGE)
    if outside > 0:
        ax.text(0.99, 0.55, f"{outside:.1f} % of decoded bins outside the frame",
                transform=ax.transAxes, ha="right", fontsize=9, color="#5A5A5A")
    ax.set_xlabel("Position (cm)")
    ax.set_ylabel("% of test bins")
    ax.set_title(f"{display} · fold-median R² = {r2:.2f} · n = {len(y_true):,} test bins "
                 "(5 folds pooled)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out = PLOT_DIR / f"{display}_decoded_vs_true_distribution.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    plt.close(fig)
    return out


def correct_split_figure(display: str) -> Path:
    y_true, y_pred, r2 = pooled_cm(display)
    edges = np.arange(X_RANGE[0], X_RANGE[1] + BIN_CM, BIN_CM)
    pct = 100.0 / len(y_true)
    ok = np.abs(y_pred - y_true) <= TOLERANCE_CM
    mu = y_true.mean()
    near = np.abs(y_pred - mu) <= 6
    color = COLOR[studies[display].cohort]

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    ax.axvspan(CONTEXT_START_CM, CONTEXT_END_CM, color="#FFF3C4", lw=0, zorder=0)
    ax.hist([y_pred[ok], y_pred[~ok]], bins=edges, weights=[np.full(ok.sum(), pct),
            np.full((~ok).sum(), pct)], stacked=True, color=[color, "#D9D9D9"],
            label=[f"decoded, correct (|error| ≤ {TOLERANCE_CM:g} cm)", "decoded, incorrect"])
    ax.hist(y_true, bins=edges, weights=np.full(len(y_true), pct), histtype="step",
            color="#5A5A5A", lw=1.2, label="true position")
    ax.axvline(mu, color="#1E1E1E", lw=1.4, ls=(0, (5, 3)), label=f"mean true position ({mu:.1f} cm)")
    ax.set_xlim(*X_RANGE)
    ax.set_xlabel("Decoded position (cm)")
    ax.set_ylabel("% of test bins")
    ax.set_title(f"{display} · correct: {100 * ok.mean():.0f} % of all predictions, "
                 f"{100 * ok[near].mean():.0f} % of those within ±6 cm of the mean "
                 f"(R² = {r2:.2f})", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out = PLOT_DIR / f"{display}_decoded_correct_split.svg"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=130)
    plt.close(fig)
    print(f"{display}: correct {100 * ok.mean():.1f} % overall; near the mean "
          f"{100 * ok[near].mean():.1f} % of {near.sum():,} predictions; elsewhere "
          f"{100 * ok[~near].mean():.1f} % of {(~near).sum():,}")
    return out


# %%
for display in RECORDINGS:
    print("wrote", distribution_figure(display).relative_to(ROOT))
    print("wrote", correct_split_figure(display).relative_to(ROOT))
