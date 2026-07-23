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
# # Preprocessing figures (presentation)
#
# Three figures illustrating the cleaning steps of the pipeline, on real data,
# using the SAME removal logic as `ibioml.preprocess_data` (imported, not
# re-implemented) so the figures match `data/processed/preprocess_report.md`.
#
# 1. **Trial selection by performance** — trials with `dPrime <= threshDPrime`
#    (or NaN) are dropped entirely.
# 2. **Neuron quality control** — mean firing rate vs drift index, with the two
#    rejection thresholds (`firingMinimo`, drift 0.5). Firing rate and drift are
#    computed AFTER time-bin removal, exactly as the pipeline does.
# 3. **Time-bin cleaning** — teleport (trial-transition) windows and immobility
#    periods (`< thVel` for `> thDur`) shaded on the real position/velocity
#    timeseries.
#
# Default dataset: Pir_N31_expert (23 neurons; expert -> threshDPrime=2.0).

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "ibioml"))

from analysis.loaders import position_in_cm  # noqa: E402
from ibioml.preprocess_data import (  # noqa: E402
    get_idx_by_low_performance,
    load_data,
    periods_immobility,
)

DATA_PATH = Path(os.environ.get(
    "DATASET",
    str(ROOT / "data" / "raw" / "piriform" / "Pir_N31_expert_bins200ms.mat"),
))
# Pipeline config (see scripts/preprocess_all.py / preprocess_report.md).
BINS_BEFORE, BINS_AFTER = 2, 2
BIN_MS = 200
BIN_S = BIN_MS / 1000
TH_VEL, TH_DUR = 1, 4
FIRING_MIN = 0.1
THRESH_DPRIME = 2.0 if "expert" in DATA_PATH.stem else -1.1
# Window (consecutive trials) for the time-bin-cleaning figure.
WIN_TRIALS = (int(os.environ.get("WIN_START", "97")), int(os.environ.get("WIN_STOP", "100")))

OUT_DIR = ROOT / "results" / "exploration" / "03-preprocessing-figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Palette (shared with 02-data-structure-figure).
INK = "#242433"
POS_COLOR = "#3B6FB0"
VEL_COLOR = "#8E6C9E"
KEEP = "#3B6FB0"
REJECT = "#CC4B4B"
TELEPORT_C = "#E0912F"
IMMOB_C = "#CC4B4B"
MUTED = "#5f6368"
GRID = "#9aa0a6"
INK_TITLE = "#202124"


def style_axis(ax, xgrid=True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    if xgrid:
        ax.grid(axis="x", color=GRID, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def compute_removed_indices(neural_data, dPrime, trialFinalBin, y, vels_binned):
    """Replicate the pipeline's time-bin removal; return the union of indices."""
    n = neural_data.shape[0]
    first, last = np.arange(BINS_BEFORE), np.arange(n - BINS_AFTER, n)
    immob, trans = periods_immobility(vels_binned, TH_VEL, BIN_MS, TH_DUR,
                                      trialFinalBin, BINS_BEFORE, BINS_AFTER)
    idx_immob = np.concatenate((first, immob, trans, last)).astype(int)
    idx_lowperf = get_idx_by_low_performance(dPrime, trialFinalBin, THRESH_DPRIME).astype(int)
    rmv_nan = np.where(np.isnan(y[:, 0]))[0].astype(int)
    union = np.union1d(rmv_nan, np.union1d(idx_immob, idx_lowperf)).astype(int)
    return union, np.array(sorted(set(immob))), np.array(sorted(set(trans)))


def drift_index(neural_data, num_blocks=2):
    """Per-neuron drift index |m0-m1|/(m0+m1) across `num_blocks` halves."""
    up_to = neural_data.shape[0] - (neural_data.shape[0] % num_blocks)
    block = up_to // num_blocks
    m = neural_data[:up_to].reshape(num_blocks, block, -1).mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(m[0] - m[1]) / (m[0] + m[1])


def suptitle(fig, text, y=0.97):
    region, animal = DATA_PATH.stem.split("_")[0], DATA_PATH.stem.split("_")[1]
    fig.suptitle(f"{region} · animal {animal} · {text}",
                 fontsize=13, fontweight="bold", x=0.06, ha="left", y=y)


# %% ---------------------------------------------------------------- Figure 1
def fig_dprime(dPrime):
    trials = np.arange(len(dPrime))
    lo, hi = np.nanmin(dPrime) - 0.12, np.nanmax(dPrime) + 0.12
    rem_thr = np.where(dPrime <= THRESH_DPRIME)[0]
    rem_nan = np.where(np.isnan(dPrime))[0]
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    ax.set_ylim(lo, hi)
    ax.set_xlim(-2, len(dPrime) + 1)
    ax.axhspan(lo, THRESH_DPRIME, color=REJECT, alpha=0.07)
    ax.plot(trials, dPrime, color=POS_COLOR, linewidth=1.6, zorder=2)
    ax.axhline(THRESH_DPRIME, color=REJECT, linestyle="--", linewidth=1.3, zorder=1)
    ax.scatter(rem_thr, dPrime[rem_thr], color=REJECT, s=32, zorder=3,
               label=f"removed — d′ ≤ {THRESH_DPRIME:g}  ({len(rem_thr)})")
    if len(rem_nan):
        ax.scatter(rem_nan, np.full(len(rem_nan), lo + 0.04), color=REJECT, marker="x",
                   s=46, linewidth=2.0, zorder=3,
                   label=f"removed — d′ undefined  ({len(rem_nan)})")
    ax.text(len(dPrime), THRESH_DPRIME, f" threshold = {THRESH_DPRIME:g}",
            color=REJECT, fontsize=9, va="bottom", ha="right")
    ax.set_xlabel("Trial index", fontsize=10)
    ax.set_ylabel("d′ (behavioural performance)", fontsize=10)
    ax.set_title("Trial selection by performance", fontsize=11, loc="left", color=INK_TITLE, pad=6)
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    style_axis(ax, xgrid=False)
    suptitle(fig, "trial selection", y=0.98)
    fig.subplots_adjust(top=0.86, left=0.09, right=0.965, bottom=0.13)
    return fig


# %% ---------------------------------------------------------------- Figure 2
def fig_neuron_qc(neural_data_clean):
    fr = np.nanmean(neural_data_clean, axis=0) / BIN_S
    drift = drift_index(neural_data_clean)
    low = fr < FIRING_MIN
    unstable = drift > 0.5
    kept = ~(low | unstable)

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    ax.axvspan(1e-3, FIRING_MIN, color=REJECT, alpha=0.06)
    ax.axhspan(0.5, 1.05, color=TELEPORT_C, alpha=0.06)
    ax.axvline(FIRING_MIN, color=REJECT, linestyle="--", linewidth=1.2)
    ax.axhline(0.5, color=TELEPORT_C, linestyle="--", linewidth=1.2)

    ax.scatter(fr[kept], drift[kept], s=55, color=KEEP, edgecolor="white",
               linewidth=0.8, label=f"kept  ({kept.sum()})", zorder=3)
    ax.scatter(fr[unstable], drift[unstable], s=70, color=TELEPORT_C, edgecolor="white",
               linewidth=0.8, marker="s", label=f"removed — drift > 0.5  ({unstable.sum()})", zorder=4)
    ax.scatter(fr[low], drift[low], s=70, color=REJECT, edgecolor="white",
               linewidth=0.8, marker="D", label=f"removed — rate < {FIRING_MIN} Hz  ({low.sum()})", zorder=4)

    ax.set_xscale("log")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Mean firing rate (Hz, log scale)", fontsize=10)
    ax.set_ylabel("Drift index  |½₁−½₂| / (½₁+½₂)", fontsize=10)
    ax.set_title(f"Neuron quality control — {neural_data_clean.shape[1]} neurons",
                 fontsize=11, loc="left", color=INK_TITLE, pad=6)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style_axis(ax, xgrid=False)
    ax.grid(color=GRID, linestyle=":", linewidth=0.6, alpha=0.4)
    suptitle(fig, "neuron selection", y=0.98)
    fig.subplots_adjust(top=0.86, left=0.11, right=0.965, bottom=0.11)
    return fig


# %% ---------------------------------------------------------------- Figure 3
def _spans(mask):
    runs, start, in_run = [], 0, False
    for i, m in enumerate(mask):
        if m and not in_run:
            start, in_run = i, True
        elif not m and in_run:
            runs.append((start, i - 1)); in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


def _shade(ax, mask, color):
    for a, b in _spans(mask):
        ax.axvspan(a * BIN_S, (b + 1) * BIN_S, color=color, alpha=0.22, linewidth=0, zorder=0)


def fig_timebins(mat, trialFinalBin, immob, trans):
    final = trialFinalBin
    t0, t1 = WIN_TRIALS
    b0 = 0 if t0 == 0 else int(final[t0 - 1] + 1)
    b1 = int(final[t1])
    win = np.arange(b0, b1 + 1)
    t = (win - b0) * BIN_S
    pos = position_in_cm(np.ravel(mat["position"])[b0:b1 + 1].astype(float))
    vel = np.ravel(mat["velocity"])[b0:b1 + 1].astype(float)
    immob_mask = np.isin(win, immob)
    trans_mask = np.isin(win, trans)

    fig, (ax_p, ax_v) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1], "hspace": 0.16})
    for ax in (ax_p, ax_v):
        _shade(ax, immob_mask, IMMOB_C)
        _shade(ax, trans_mask, TELEPORT_C)

    ax_p.plot(t, pos, color=POS_COLOR, linewidth=1.9)
    ax_p.set_ylabel("Position (cm)", fontsize=10)
    ax_p.set_title("Position in the corridor", fontsize=11, loc="left", color=INK_TITLE, pad=6)
    ax_p.set_ylim(0, pos.max() * 1.08)
    style_axis(ax_p)

    ax_v.axhline(TH_VEL, color=MUTED, linestyle="--", linewidth=0.9, alpha=0.8)
    ax_v.text(t[-1], TH_VEL, f" immobility threshold ({TH_VEL} cm/s)", color=MUTED,
              fontsize=8, va="bottom", ha="right")
    ax_v.plot(t, vel, color=VEL_COLOR, linewidth=1.6)
    ax_v.set_ylabel("Velocity (cm/s)", fontsize=10)
    ax_v.set_title("Velocity", fontsize=11, loc="left", color=INK_TITLE, pad=6)
    ax_v.set_xlabel("Time (s)", fontsize=10)
    ax_v.set_xlim(t[0], t[-1] + BIN_S)
    style_axis(ax_v)

    legend = [Patch(facecolor=TELEPORT_C, alpha=0.5, label="removed — trial transition (teleport)"),
              Patch(facecolor=IMMOB_C, alpha=0.5, label=f"removed — immobility (> {TH_DUR} s under {TH_VEL} cm/s)")]
    ax_p.legend(handles=legend, frameon=False, fontsize=9, loc="upper left", ncol=1)
    suptitle(fig, f"time-bin cleaning · trials {t0}–{t1}", y=0.975)
    fig.subplots_adjust(top=0.885, left=0.075, right=0.965, bottom=0.10)
    return fig


# %%
if __name__ == "__main__":
    mat, neural_data, rewCtxt, trialFinalBin, dPrime, criterion, rewOdor, pos_b, vels_b = load_data(str(DATA_PATH))
    y = np.concatenate((pos_b, vels_b), axis=1)
    union, immob, trans = compute_removed_indices(neural_data, dPrime, trialFinalBin, y, vels_b)
    neural_data_clean = np.delete(neural_data, union, 0)
    print(f"\ntime bins: {neural_data.shape[0]} -> kept {neural_data_clean.shape[0]} "
          f"({100 * (1 - neural_data_clean.shape[0] / neural_data.shape[0]):.1f}% removed)")

    stem = DATA_PATH.stem
    figures = {
        f"{stem}_1-dprime.png": fig_dprime(dPrime),
        f"{stem}_2-neuron-qc.png": fig_neuron_qc(neural_data_clean),
        f"{stem}_3-timebins.png": fig_timebins(mat, trialFinalBin, immob, trans),
    }
    for name, fig in figures.items():
        fig.savefig(OUT_DIR / name, dpi=300)
        print(f"saved {OUT_DIR / name}")
