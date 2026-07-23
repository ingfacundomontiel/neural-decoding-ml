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
# # Data-structure figure (presentation)
#
# A single-trial view of the data the decoding pipeline consumes, in three
# layers that share one time axis (seconds within the trial):
#
# 1. **Neural population activity** — every simultaneously recorded neuron as a
#    row; each 200 ms bin with `k` spikes is drawn as `k` ticks spread across
#    the bin (density therefore reflects firing rate). This is the *input* (X).
# 2. **Position in the corridor (cm)** — one of the decoding *targets* (y).
# 3. **Velocity** — the other decoding target.
#
# The script writes a **combined** figure and each layer as a **standalone**
# image (PNG only), so panels can be placed individually on slides.
#
# Honesty note: the raw `.mat` stores spikes already binned at 200 ms, so we do
# not have individual spike times. The per-bin count is rendered as evenly
# spaced ticks inside its bin — a standard way to show binned counts as a
# raster, not a claim of sub-bin timing.
#
# Configurable via env vars:
#   DATASET  path to a recording .mat   (default Pir_L5_expert_bins200ms)
#   TRIAL    trial index to plot        (default 77 — clean corridor run)

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from analysis.loaders import CORRIDOR_END_CM, load_mat, position_in_cm  # noqa: E402

DATA_PATH = Path(os.environ.get(
    "DATASET",
    str(ROOT / "data" / "raw" / "piriform" / "Pir_L5_expert_bins200ms.mat"),
))
TRIAL = int(os.environ.get("TRIAL", "77"))
BIN_S = 0.2  # 200 ms bins

# Draw the dashed corridor-end reference line + label on the position panel.
# Set SHOW_CORRIDOR_END=0 for a cleaner panel (e.g. slides).
SHOW_CORRIDOR_END = os.environ.get("SHOW_CORRIDOR_END", "1") != "0"

OUT_DIR = ROOT / "results" / "exploration" / "02-data-structure-figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colours — one hue per single-series panel (no dual axes, no cycling).
INK = "#242433"          # raster ticks (near-black, warm)
POS_COLOR = "#3B6FB0"    # position — blue
VEL_COLOR = "#8E6C9E"    # velocity — muted purple
GRID = "#9aa0a6"
MUTED = "#5f6368"


def trial_bounds(mat: dict, trial: int) -> tuple[int, int]:
    """Return (start_bin, stop_bin_exclusive) for `trial` (0-indexed)."""
    final = (mat["trialFinalBin"] - 1).astype(int)  # MATLAB 1-indexed
    start = 0 if trial == 0 else int(final[trial - 1] + 1)
    return start, int(final[trial] + 1)


def trial_arrays(mat: dict, trial: int):
    """Slice a trial and return (t, activity, position_cm, velocity)."""
    start, stop = trial_bounds(mat, trial)
    t = (np.arange(start, stop) - start) * BIN_S
    activity = mat["neuronActivity"][start:stop].astype(int)
    position = position_in_cm(mat["position"][start:stop].astype(float))
    velocity = mat["velocity"][start:stop].astype(float)
    return t, activity, position, velocity


def raster_segments(activity: np.ndarray) -> tuple[list, int]:
    """Build tick line-segments for a (n_bins, n_neurons) spike-count block.

    Neurons are sorted by total activity (busiest at the top). Each bin with
    `k` spikes contributes `k` ticks evenly spread across the bin's 200 ms.
    Returns (segments, n_neurons) for a LineCollection.
    """
    order = np.argsort(activity.sum(axis=0))  # ascending -> busiest ends on top
    activity = activity[:, order]
    n_bins, n_neurons = activity.shape
    tick_h = 0.4  # half-height of a tick, in neuron-row units
    segments = []
    for b in range(n_bins):
        for row in range(n_neurons):
            k = int(activity[b, row])
            if k == 0:
                continue
            xs = (b + (np.arange(k) + 0.5) / k) * BIN_S
            y0, y1 = row + 1 - tick_h, row + 1 + tick_h
            segments.extend([[(x, y0), (x, y1)] for x in xs])
    return segments, n_neurons


def style_axis(ax) -> None:
    """Shared cosmetic treatment: hide top/right spines, dotted x-grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)


def draw_raster(ax, t, activity) -> None:
    segments, n_neurons = raster_segments(activity)
    lw = 0.9 if n_neurons <= 30 else 0.7
    ax.add_collection(LineCollection(segments, colors=INK, linewidths=lw))
    ax.set_ylim(0.3, n_neurons + 0.7)
    ax.set_xlim(t[0], t[-1] + BIN_S)
    ax.set_ylabel("Neurons\n(sorted by activity)", fontsize=10)
    ax.set_yticks([1, n_neurons])
    ax.set_title(f"Neural population activity  —  {n_neurons} simultaneously recorded neurons",
                 fontsize=11, loc="left", color="#202124", pad=6)
    style_axis(ax)


def draw_position(ax, t, position) -> None:
    ax.fill_between(t, position, color=POS_COLOR, alpha=0.12)
    ax.plot(t, position, color=POS_COLOR, linewidth=2.0)
    if SHOW_CORRIDOR_END:
        ax.axhline(CORRIDOR_END_CM, color=MUTED, linestyle="--", linewidth=0.9, alpha=0.6)
        ax.text(t[-1] + BIN_S, CORRIDOR_END_CM - 1.5, f"corridor end ({CORRIDOR_END_CM:.0f} cm)",
                fontsize=8, color=MUTED, va="top", ha="right")
    ax.set_xlim(t[0], t[-1] + BIN_S)
    ax.set_ylim(0, max(position.max() * 1.08, CORRIDOR_END_CM * 1.1))
    ax.set_ylabel("Position (cm)", fontsize=10)
    ax.set_title("Position in the corridor", fontsize=11, loc="left", color="#202124", pad=6)
    style_axis(ax)


def draw_velocity(ax, t, velocity) -> None:
    ax.axhline(0, color=MUTED, linewidth=0.7, alpha=0.5)
    ax.fill_between(t, velocity, color=VEL_COLOR, alpha=0.12)
    ax.plot(t, velocity, color=VEL_COLOR, linewidth=2.0)
    ax.set_xlim(t[0], t[-1] + BIN_S)
    ax.set_ylabel("Velocity", fontsize=10)
    ax.set_title("Velocity", fontsize=11, loc="left", color="#202124", pad=6)
    style_axis(ax)


def suptitle(fig, trial: int, y: float) -> None:
    region, animal = DATA_PATH.stem.split("_")[0], DATA_PATH.stem.split("_")[1]
    fig.suptitle(f"{region} · animal {animal} · trial {trial}",
                 fontsize=14, fontweight="bold", x=0.065, ha="left", y=y)


def build_combined(t, activity, position, velocity, trial):
    fig, (ax_r, ax_p, ax_v) = plt.subplots(
        3, 1, figsize=(11, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.0, 1.0], "hspace": 0.18},
    )
    draw_raster(ax_r, t, activity)
    draw_position(ax_p, t, position)
    draw_velocity(ax_v, t, velocity)
    ax_v.set_xlabel("Time within trial (s)", fontsize=10)
    suptitle(fig, trial, y=0.97)
    fig.subplots_adjust(top=0.915, left=0.10, right=0.955, bottom=0.075)
    return fig


def build_panel(t, data, trial, draw, figsize):
    """One standalone panel via `draw(ax, t, data)`."""
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax, t, data)
    ax.set_xlabel("Time within trial (s)", fontsize=10)
    suptitle(fig, trial, y=0.965)
    fig.subplots_adjust(top=0.80, left=0.10, right=0.955, bottom=0.16)
    return fig


# %%
if __name__ == "__main__":
    mat = load_mat(DATA_PATH)
    t, activity, position, velocity = trial_arrays(mat, TRIAL)
    stem = f"{DATA_PATH.stem}_trial{TRIAL}"

    figures = {
        f"{stem}.png": build_combined(t, activity, position, velocity, TRIAL),
        f"{stem}_panel1-activity.png": build_panel(t, activity, TRIAL, draw_raster, (11, 4.3)),
        f"{stem}_panel2-position.png": build_panel(t, position, TRIAL, draw_position, (11, 3.2)),
        f"{stem}_panel3-velocity.png": build_panel(t, velocity, TRIAL, draw_velocity, (11, 3.2)),
    }
    for name, fig in figures.items():
        fig.savefig(OUT_DIR / name, dpi=300)
        print(f"saved {OUT_DIR / name}")
