"""Summary figures: dataset scatter, per-recording R² boxplots, position-vs-speed R².

Sizes are provisional slot sizes (cm) until the layout is budgeted; change the
constants and re-run — fonts stay at their printed size.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import poster_data as pdata
from poster_style import (INK, INK_SOFT, REGION_COLOR, REGION_LIGHT, REGION_NAME,
                          REGION_ORDER, cm, save, setup)

import matplotlib.pyplot as plt

SIZE_SCATTER = (18.5, 11.5)
# Poster-level key (region colours, hollow = naive) replaces per-figure legends.
MARKER_S = 260          # scatter marker area (pt²)
LABEL_FS = 20
DEFAULT_OFFSET = (9, 6)
OFFSETS_DATASETS = {"Pir-Naive-N53": (9, 4), "DG-Naive-S38": (-20, -26), "CA3-Naive-S31": (-44, -6),
                    "Pir-Expert-L3": (-34, 8), "DG-Expert-S19": (-14, 12), "DG-Expert-S17": (6, 10),
                    "CA3-Expert-S18": (-14, -28), "CA3-Expert-S17": (6, -26), "Pir-Expert-N31": (-14, -28)}
# Hand-placed label centres (data coords) for the position-vs-speed scatter; the rest are auto-placed.
LABELS_POSVEL = {"CA3-Naive-S31": (0.20, 0.30), "CA3-Expert-S17": (0.27, 0.37),
                 "CA3-Expert-S18": (0.69, 0.47)}
# Points far enough from their neighbours get a plain label next to them (no leader line):
# offset in points from the marker centre + text alignment.
PLAIN_POSVEL = {"Pir-Naive-L7": (-12, 6, "right", "bottom"), "Pir-Naive-N53": (-12, 6, "right", "bottom"),
                "Pir-Expert-L5": (-12, 6, "right", "bottom"), "Pir-Expert-N31": (13, 0, "left", "center"),
                "DG-Expert-S17": (13, 0, "left", "center"), "DG-Expert-S19": (13, 0, "left", "center"),
                "DG-Expert-S18": (13, 0, "left", "center"), "CA3-Naive-S30": (0, -13, "center", "top")}
Y_FLOOR = -0.1
SIZE_BOX = (38.0, 11.5)
GROUP_GAP = 1.0


def load_all(target: str) -> list[pdata.Run]:
    with contextlib.redirect_stdout(io.StringIO()):  # ibioml filters print per call
        return [pdata.load_run(i.stem, target) for i in pdata.studies(target)]


def marker_style(region: str, cohort: str) -> dict:
    face = REGION_COLOR[region] if cohort == "expert" else "white"
    return {"marker": "o", "s": MARKER_S, "facecolor": face, "edgecolor": REGION_COLOR[region],
            "linewidth": 2.5, "zorder": 3}


def cohort_legend_handles(region: str = "Pir") -> list:
    c = REGION_COLOR[region]
    return [Line2D([], [], marker="o", ls="", ms=14, mfc="white", mec=c, mew=2.5, label="naive"),
            Line2D([], [], marker="o", ls="", ms=14, mfc=c, mec=c, mew=2.5, label="expert")]


def region_legend_handles() -> list:
    return [Patch(color=REGION_COLOR[r], label=REGION_NAME[r]) for r in REGION_ORDER]


def place_labels(fig, ax, items: list[tuple[float, float, str, str, str]],
                 fixed: dict[str, tuple[float, float]] | None = None,
                 plain: dict[str, tuple[float, float, str, str]] | None = None) -> None:
    """Spread point labels so none overlaps a point or another label; a straight
    leader line in the point's colour joins each label to its point.

    Greedy search in display space: candidate centres on rings around each point
    (densest points placed first); the cost penalises leaving the axes, overlapping
    placed labels or markers, and distance from the point. Labels whose key is in
    *fixed* go exactly at the given data-coordinate centre and are placed first;
    labels in *plain* sit next to their point (offset in points), without a line.
    """
    fixed, plain = fixed or {}, plain or {}
    fig.canvas.draw()
    to_px = ax.transData.transform
    frame = ax.get_window_extent()
    px_per_pt = fig.dpi / 72
    r_marker = np.sqrt(MARKER_S / np.pi) * px_per_pt
    pts = np.array([to_px((x, y)) for x, y, _, _, _ in items])
    sizes = []
    for _, _, text, _, _ in items:
        t = ax.text(0, 0, text, fontsize=LABEL_FS)
        bb = t.get_window_extent(fig.canvas.get_renderer())
        sizes.append((bb.width + 6, bb.height + 2))
        t.remove()
    density = [np.sum(np.hypot(*(pts - p).T) < 6 * r_marker) for p in pts]
    placed: list[tuple[float, float, float, float]] = []
    centres = {}
    renderer = fig.canvas.get_renderer()
    for x, y, text, color, key in items:
        if key in plain:
            dx, dy, ha, va = plain[key]
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize=LABEL_FS, color=color, zorder=5, annotation_clip=False)
            bb = t.get_window_extent(renderer)
            placed.append((bb.x0, bb.y0, bb.x1, bb.y1))
    for i, item in enumerate(items):
        if item[4] in plain:
            continue
        if item[4] in fixed:
            cx, cy = to_px(fixed[item[4]])
            w, h = sizes[i]
            centres[i] = (cx, cy)
            placed.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    todo = [k for k in range(len(items)) if k not in centres and items[k][4] not in plain]
    for i in sorted(todo, key=lambda k: -density[k]):
        w, h = sizes[i]
        best, best_cost = None, np.inf
        for radius in (1.5, 2.2, 3.0, 4.0, 5.2):
            for ang in np.deg2rad(np.arange(0, 360, 15)):
                cx = pts[i, 0] + np.cos(ang) * (r_marker + radius * h)
                cy = pts[i, 1] + np.sin(ang) * (r_marker + radius * h)
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                cost = radius
                if box[0] < frame.x0 or box[2] > frame.x1 or box[1] < frame.y0 or box[3] > frame.y1:
                    cost += 1e4
                for q in placed:
                    ox = max(0, min(box[2], q[2]) - max(box[0], q[0]))
                    oy = max(0, min(box[3], q[3]) - max(box[1], q[1]))
                    cost += 50 * ox * oy / (w * h)
                for p in pts:
                    dx = max(box[0] - p[0], 0, p[0] - box[2])
                    dy = max(box[1] - p[1], 0, p[1] - box[3])
                    if np.hypot(dx, dy) < r_marker * 1.15:
                        cost += 30
                if cost < best_cost:
                    best, best_cost = (cx, cy, box), cost
        centres[i] = best[:2]
        placed.append(best[2])
    to_data = ax.transData.inverted().transform
    for i, (x, y, text, color, _) in enumerate(items):
        if i not in centres:
            continue
        ax.annotate(text, (x, y), xytext=to_data(centres[i]), textcoords="data", annotation_clip=False,
                    ha="center", va="center", fontsize=LABEL_FS, color=color, zorder=5,
                    bbox={"boxstyle": "square,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.85},
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 1.6,
                                "shrinkA": 0, "shrinkB": np.sqrt(MARKER_S / np.pi) + 1})


def fig_datasets(runs: list[pdata.Run]) -> None:
    fig, ax = plt.subplots(figsize=cm(*SIZE_SCATTER), layout="constrained")
    for r in runs:
        n_neurons, n_trials = r.n_features - 2, len(np.unique(r.trial))
        ax.scatter(n_trials, n_neurons, **marker_style(r.info.region, r.info.cohort))
        ax.annotate(r.info.animal, (n_trials, n_neurons),
                    xytext=OFFSETS_DATASETS.get(r.info.display, DEFAULT_OFFSET),
                    textcoords="offset points", fontsize=LABEL_FS, color=INK_SOFT)
    ax.set_xlabel("Trials")
    ax.set_ylabel("Neurons")
    ax.set_ylim(0, None)
    save(fig, "datasets_neurons_vs_trials")


def box_positions(runs: list[pdata.Run]) -> np.ndarray:
    xs, x, prev = [], 0.0, None
    for r in runs:
        if prev is not None and r.info.region != prev:
            x += GROUP_GAP
        xs.append(x)
        x += 1
        prev = r.info.region
    return np.array(xs)


def fig_boxplot(runs: list[pdata.Run], target: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=cm(*SIZE_BOX), layout="constrained")
    xs = box_positions(runs)
    rng = np.random.default_rng(0)
    for x, r in zip(xs, runs, strict=True):
        reg, expert = r.info.region, r.info.cohort == "expert"
        face = REGION_COLOR[reg] if expert else REGION_LIGHT[reg]
        ax.boxplot([r.fold_r2], positions=[x], widths=0.62, patch_artist=True,
                   showfliers=False, whis=(0, 100),
                   boxprops={"facecolor": face, "edgecolor": REGION_COLOR[reg], "lw": 2},
                   medianprops={"color": "white" if expert else INK, "lw": 3},
                   whiskerprops={"color": REGION_COLOR[reg], "lw": 2},
                   capprops={"color": REGION_COLOR[reg], "lw": 2})
        folds = np.array(r.fold_r2)
        inside = folds >= Y_FLOOR
        ax.scatter(x + rng.uniform(-0.12, 0.12, inside.sum()), folds[inside], s=40,
                   color=INK, zorder=4, lw=0)
        for val in folds[~inside]:  # clipped fold: arrow at the floor + its value
            ax.annotate(f"{val:.2f}", (x, Y_FLOOR), xytext=(x + 0.18, Y_FLOOR + 0.06),
                        fontsize=18, color=INK_SOFT, ha="left", va="center",
                        arrowprops={"arrowstyle": "-|>", "color": INK_SOFT, "lw": 1.5},
                        annotation_clip=False)
        ax.text(x, max(r.fold_r2) + 0.03, f"{r.median_r2:.2f}", ha="center", va="bottom",
                fontsize=20, color=INK)
    ax.axhline(0, color=INK_SOFT, lw=1.4, ls=(0, (4, 3)), zorder=0)
    ax.set_xticks(xs, [r.info.animal for r in runs], fontsize=20)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    top = max(max(r.fold_r2) for r in runs)
    ax.set_ylim(Y_FLOOR, top + 0.12)
    ax.set_ylabel(ylabel)
    for reg in REGION_ORDER:
        sel = [x for x, r in zip(xs, runs, strict=True) if r.info.region == reg]
        ax.text(np.mean(sel), -0.16, REGION_NAME[reg], transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=24, fontweight="semibold", color=REGION_COLOR[reg])
    save(fig, f"boxplot_r2_{target}")


def fig_pos_vs_vel(pos: list[pdata.Run], vel: list[pdata.Run]) -> None:
    fig, ax = plt.subplots(figsize=cm(*SIZE_SCATTER), layout="constrained")
    lim = (0, 0.7)
    ax.plot(lim, lim, color=INK_SOFT, lw=1.4, ls=(0, (4, 3)), zorder=0)
    items = []
    for p, v in zip(pos, vel, strict=True):
        assert p.info.stem == v.info.stem
        ax.scatter(v.median_r2, p.median_r2, **marker_style(p.info.region, p.info.cohort))
        items.append((v.median_r2, p.median_r2, p.info.animal, REGION_COLOR[p.info.region],
                      p.info.display))
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Speed decoding R²")
    ax.set_ylabel("Position decoding R²")
    place_labels(fig, ax, items, fixed=LABELS_POSVEL, plain=PLAIN_POSVEL)
    save(fig, "scatter_r2_pos_vs_speed")


def main() -> None:
    setup()
    pos, vel = load_all("pos"), load_all("vel")
    fig_datasets(pos)
    fig_boxplot(pos, "pos", "Position decoding R²")
    fig_boxplot(vel, "vel", "Speed decoding R²")
    fig_pos_vs_vel(pos, vel)


if __name__ == "__main__":
    main()
