"""Predictions over real data — one figure per region, naive | expert side by side.

Rule for the examples: the best recording per region × cohort (highest median
fold R²), and within it the test fold whose R² is closest to that median. The
x axis is the fold's test bins concatenated in time: bins removed by the
preprocessing are absent, so it is not wall-clock time.
All panels share the same size and axis ranges across regions.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np

import poster_data as pdata
from poster_style import (INK, REGION_COLOR, REGION_ORDER, cm, save, setup, shade_context)

import matplotlib.pyplot as plt

SIZE_ROW = (38.0, 6.3)
WINDOW_S = 200.0
BIN_S = 0.2
YLIM = {"pos": (-8, 72), "vel": (-6, 30)}
YLABEL = {"pos": "Position\n(cm)", "vel": "Speed\n(cm/s)"}


def best_per_group(target: str) -> dict[tuple[str, str], pdata.Run]:
    with contextlib.redirect_stdout(io.StringIO()):
        runs = [pdata.load_run(i.stem, target) for i in pdata.studies(target)]
    best: dict[tuple[str, str], pdata.Run] = {}
    for r in runs:
        key = (r.info.region, r.info.cohort)
        if key not in best or r.median_r2 > best[key].median_r2:
            best[key] = r
    return best


def median_fold(run: pdata.Run) -> int:
    """1-based id of the fold whose R² is closest to the recording's median."""
    return int(np.argmin(np.abs(np.array(run.fold_r2) - run.median_r2))) + 1


def trace_panel(ax, run: pdata.Run, target: str, show_ylabel: bool, show_xlabel: bool) -> None:
    k = median_fold(run)
    sel = run.fold_id == k
    n = int(WINDOW_S / BIN_S)
    y_true, y_pred = run.y_true[sel][:n], run.y_pred[sel][:n]
    t = np.arange(len(y_true)) * BIN_S
    if target == "pos":
        shade_context(ax, orientation="y")
    ax.plot(t, y_true, color=INK, lw=2.2, label="real")
    ax.plot(t, y_pred, color=REGION_COLOR[run.info.region], lw=2.2, label="decoded")
    ax.set_xlim(0, WINDOW_S)
    ax.set_ylim(*YLIM[target])
    if show_xlabel:
        ax.set_xlabel("Time (s)")
    if show_ylabel:
        ax.set_ylabel(YLABEL[target])
    ax.set_title(f"{run.info.animal} · {run.info.cohort}", loc="left", fontsize=22,
                 fontweight="semibold")
    ax.text(1.0, 1.02, f"R² = {run.fold_r2[k - 1]:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=22)


def fig_region(best: dict, region: str, target: str) -> None:
    last = region == REGION_ORDER[-1]
    fig, axes = plt.subplots(1, 2, figsize=cm(*SIZE_ROW), layout="constrained", sharey=True)
    for ax, cohort in zip(axes, ("naive", "expert"), strict=True):
        trace_panel(ax, best[(region, cohort)], target, show_ylabel=cohort == "naive",
                    show_xlabel=last)
    save(fig, f"pred_{target}_{region}")


def main() -> None:
    setup()
    for target in ("pos", "vel"):
        best = best_per_group(target if target == "pos" else "vel")
        if target == "vel":  # speed examples: the same recordings as position
            pos_best = best_per_group("pos")
            with contextlib.redirect_stdout(io.StringIO()):
                best = {k: pdata.load_run(r.info.stem, "vel") for k, r in pos_best.items()}
        for region in REGION_ORDER:
            fig_region(best, region, target)


if __name__ == "__main__":
    main()
