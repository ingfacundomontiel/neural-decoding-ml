"""Decoded vs true position along the corridor — one column per region.

Top: conditional heatmap P(predicted | true) of the best expert recording of the
region (each 2 cm column normalised by its own count; columns with < MIN_COUNT test
bins blank), identity line, context-zone edges. Bottom: MAE per 2 cm bin of that
expert (solid) and of the best naive recording of the same region (dashed).
Test predictions of the five outer folds pooled (one out-of-sample prediction per
kept bin). Same logic as exploration/08, restyled for the poster.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

import poster_data as pdata
from poster_style import (CONTEXT_ENTRY_CM, CONTEXT_EXIT_CM, INK, INK_SOFT, REGION_COLOR,
                          REGION_LIGHT, REGION_NAME, REGION_ORDER, cm, save, setup, shade_context)

import matplotlib.pyplot as plt

SIZE = (38.0, 16.0)
BIN = 2.0
X_EDGES = np.arange(0, 64 + BIN, BIN)
Y_EDGES = np.arange(-6, 70 + BIN, BIN)
MIN_COUNT = 30
P99_AT = 0.8   # the 99th percentile of P(decoded | true) sits at 80 % of the colour ramp


def best_runs() -> dict[tuple[str, str], pdata.Run]:
    with contextlib.redirect_stdout(io.StringIO()):
        runs = [pdata.load_run(i.stem, "pos") for i in pdata.studies("pos")]
    best: dict[tuple[str, str], pdata.Run] = {}
    for r in runs:
        key = (r.info.region, r.info.cohort)
        if key not in best or r.median_r2 > best[key].median_r2:
            best[key] = r
    return best


def conditional_heatmap(run: pdata.Run) -> np.ndarray:
    h, _, _ = np.histogram2d(run.y_true, run.y_pred, bins=(X_EDGES, Y_EDGES))
    counts = h.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = h / counts
    p[counts[:, 0] < MIN_COUNT] = np.nan
    return p.T


def binned_mae(run: pdata.Run) -> np.ndarray:
    idx = np.digitize(run.y_true, X_EDGES) - 1
    err = np.abs(run.y_pred - run.y_true)
    return np.array([err[idx == i].mean() if (idx == i).sum() >= MIN_COUNT else np.nan
                     for i in range(len(X_EDGES) - 1)])


def heat_norm(experts: list[pdata.Run]) -> PowerNorm:
    """One scale for the three panels: the max reaches the full colour (nothing clips);
    a power law puts the pooled 99th percentile at P99_AT of the ramp."""
    vals = np.concatenate([conditional_heatmap(r).ravel() for r in experts])
    vmax, p99 = np.nanmax(vals), np.nanpercentile(vals, 99)
    return PowerNorm(gamma=np.log(P99_AT) / np.log(p99 / vmax), vmin=0, vmax=vmax)


def column(fig, gs, j: int, region: str, expert: pdata.Run, naive: pdata.Run, norm: PowerNorm):
    first = j == 0
    ax_h = fig.add_subplot(gs[0, j])
    ax_m = fig.add_subplot(gs[1, j], sharex=ax_h)
    cmap = LinearSegmentedColormap.from_list(region, ["#FFFFFF", REGION_COLOR[region]])
    cmap.set_bad("white")
    ax_h.imshow(conditional_heatmap(expert), origin="lower", aspect="auto", cmap=cmap,
                norm=norm, extent=(X_EDGES[0], X_EDGES[-1], Y_EDGES[0], Y_EDGES[-1]),
                interpolation="nearest")
    ax_h.plot([-10, 80], [-10, 80], color=INK, lw=1.6, ls=(0, (5, 4)))
    for v in (CONTEXT_ENTRY_CM, CONTEXT_EXIT_CM):
        ax_h.axvline(v, color=INK_SOFT, lw=1.3, ls=":")
        ax_h.axhline(v, color=INK_SOFT, lw=1.3, ls=":")
    ax_h.set_xlim(0, 64)
    ax_h.set_ylim(-6, 70)
    ax_h.set_title(f"{REGION_NAME[region]} · {expert.info.animal}", loc="left", fontsize=26,
                   color=REGION_COLOR[region], fontweight="semibold")
    ax_h.tick_params(labelbottom=False)
    centers = (X_EDGES[:-1] + X_EDGES[1:]) / 2
    shade_context(ax_m)
    ax_m.plot(centers, binned_mae(expert), color=REGION_COLOR[region], lw=3.2,
              label=f"expert {expert.info.animal}")
    ax_m.plot(centers, binned_mae(naive), color=REGION_LIGHT[region], lw=3.2, ls=(0, (5, 3)),
              label=f"naive {naive.info.animal}")
    ax_m.set_ylim(0, 35)
    ax_m.set_xlabel("True position (cm)")
    ax_m.legend(loc="upper center", fontsize=19, ncols=2, handlelength=1.4, columnspacing=1.0)
    if first:
        ax_h.set_ylabel("Decoded (cm)")
        ax_m.set_ylabel("MAE (cm)")


def main() -> None:
    setup()
    best = best_runs()
    fig = plt.figure(figsize=cm(*SIZE), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1])
    norm = heat_norm([best[(r, "expert")] for r in REGION_ORDER])
    for j, region in enumerate(REGION_ORDER):
        column(fig, gs, j, region, best[(region, "expert")], best[(region, "naive")], norm)
    fig.align_ylabels()
    save(fig, "corridor_heatmap_mae")


if __name__ == "__main__":
    main()
