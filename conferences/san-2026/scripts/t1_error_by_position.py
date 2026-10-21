"""Open topic T1 — is the error-by-position shape information or shrinkage?

An MSE-trained regressor pulls uncertain predictions toward the mean of its
training targets, so its error is smallest near that mean even when the neural
information is uniform along the corridor. This script:

1. **Simulation** (uniform information by construction): position x, one noisy
   "neural" feature r = x + ε (σ constant), an MSE-trained MLP decoder. Its MAE
   is U-shaped with the minimum at the training mean; moving the training range
   (tail included vs removed) moves the minimum. The local resolution
   σ_resid / slope (below) stays flat — the check that it is shrinkage-invariant.
2. **Real data, final pipeline**: per recording and 2.5 cm position bin — MAE of
   the MLP, MAE of the no-information decoder (predicting the train-fold mean,
   i.e. |x − μ_k|), and the local resolution: in a ±5 cm window, regress
   prediction on true position; resolution = residual SD / slope (cm). Linear
   shrinkage scales slope and residual SD alike, so the ratio does not see it.

Outputs: figures/t1/*.png|svg and figures/t1/t1_zone_summary.{csv,md}.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

import poster_data as pdata
from poster_style import (CONTEXT_ENTRY_CM, CONTEXT_EXIT_CM, FIG_DIR, INK, INK_SOFT,
                          REGION_COLOR, REGION_LIGHT, REGION_NAME, REGION_ORDER, cm, setup,
                          shade_context)

import matplotlib.pyplot as plt

OUT = FIG_DIR / "t1"
EDGES = np.arange(0, 65, 2.5)
CENTERS = (EDGES[:-1] + EDGES[1:]) / 2
HALF_WIN = 5.0
MIN_ROWS = 40
ZONES = {"start 0–22": (0, 22), "mid 22–44.7": (22, CONTEXT_ENTRY_CM),
         "context 44.7–62.9": (CONTEXT_ENTRY_CM, CONTEXT_EXIT_CM + 0.01)}


def local_resolution(x: np.ndarray, pred: np.ndarray, centers=CENTERS) -> tuple[np.ndarray, np.ndarray]:
    """Per centre: (slope, resolution = residual SD / slope) of pred ~ x within ±HALF_WIN."""
    slopes, res = [], []
    for c in centers:
        m = np.abs(x - c) <= HALF_WIN
        if m.sum() < MIN_ROWS or np.ptp(x[m]) < HALF_WIN:
            slopes.append(np.nan)
            res.append(np.nan)
            continue
        b, a = np.polyfit(x[m], pred[m], 1)
        resid_sd = np.std(pred[m] - (a + b * x[m]))
        slopes.append(b)
        res.append(resid_sd / b if b > 0.05 else np.nan)
    return np.array(slopes), np.array(res)


def binned_mae(x: np.ndarray, err: np.ndarray) -> np.ndarray:
    idx = np.digitize(x, EDGES) - 1
    return np.array([np.mean(err[idx == i]) if (idx == i).sum() >= MIN_ROWS // 2 else np.nan
                     for i in range(len(CENTERS))])


# --- 1. simulation -------------------------------------------------------------

def simulate(x_max: float, sigma_out: float, sigma_ctx: float, seed: int = 0) -> dict:
    """MSE-optimal decoder E[x | r] for r = x + ε, uniform prior on [0, x_max].

    Any decoder trained to minimise squared error converges to this posterior mean,
    so it shows what shrinkage alone does. σ may be smaller inside the context zone.
    """
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, x_max, 1200)
    sig = lambda x: np.where((x >= CONTEXT_ENTRY_CM) & (x < CONTEXT_EXIT_CM),  # noqa: E731
                             sigma_ctx, sigma_out)
    r_grid = np.linspace(-4 * sigma_out, x_max + 4 * sigma_out, 3000)
    like = np.exp(-0.5 * ((r_grid[:, None] - grid[None, :]) / sig(grid)[None, :]) ** 2) / sig(grid)
    post_mean = (like * grid).sum(1) / like.sum(1)
    centers = np.arange(1.25, x_max, 2.5)
    xs = np.repeat(centers, 40000)
    pred = np.interp(xs + rng.normal(0, 1, xs.size) * sig(xs), r_grid, post_mean)
    mae = np.abs(pred - xs).reshape(len(centers), -1).mean(1)
    mean_pred = pred.reshape(len(centers), -1).mean(1)
    slope = np.gradient(mean_pred, centers)
    x_all = rng.uniform(0, x_max, 200000)
    p_all = np.interp(x_all + rng.normal(0, 1, x_all.size) * sig(x_all), r_grid, post_mean)
    r2 = 1 - np.mean((p_all - x_all) ** 2) / np.var(x_all)
    return {"centers": centers, "mae": mae, "slope": slope, "mean": x_max / 2, "r2": r2}


SCENARIOS = (  # (x_max, σ outside, σ inside context, colour, label)
    (95, 25, 25, INK_SOFT, "uniform info, range 0–95 cm (tail kept)"),
    (63, 25, 25, REGION_COLOR["Pir"], "uniform info, range 0–63 cm (tail removed)"),
    (63, 25, 10, REGION_COLOR["DG"], "range 0–63 cm, extra info inside the context"),
)


def fig_simulation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=cm(38, 13), layout="constrained")
    for x_max, s_out, s_ctx, color, label in SCENARIOS:
        s = simulate(x_max, s_out, s_ctx)
        axes[0].plot(s["centers"], s["mae"], color=color, label=f"{label} (R² {s['r2']:.2f})")
        axes[0].axvline(s["mean"], color=color, lw=1.5, ls=(0, (4, 3)))
        axes[1].plot(s["centers"], s["slope"], color=color)
    for ax in axes:
        shade_context(ax)
        ax.set_xlabel("True position (cm)")
        ax.set_xlim(0, 95)
    axes[0].set_ylabel("MAE (cm)")
    axes[0].set_ylim(0, 45)
    axes[0].set_title("MAE (dashed: training mean)", loc="left",
                      fontsize=22)
    axes[1].set_ylabel("Local slope (pred vs true)")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_title("Local slope of the predictions", loc="left",
                      fontsize=22)
    axes[0].legend(loc="upper center", fontsize=17)
    fig.savefig(OUT / "sim_shrinkage.svg")
    fig.savefig(OUT / "sim_shrinkage.png", dpi=60)
    plt.close(fig)


# --- 2. real data ----------------------------------------------------------------

def recording_curves(run: pdata.Run) -> dict:
    x, pred = run.y_true, run.y_pred
    null = np.empty_like(x)
    for k in np.unique(run.fold_id):
        null[run.fold_id == k] = np.abs(x[run.fold_id == k] - x[run.fold_id != k].mean())
    slope, res = local_resolution(x, pred)
    zones = {}
    for z, (lo, hi) in ZONES.items():
        m = (x >= lo) & (x < hi)
        s_z, r_z = local_resolution(x[m], pred[m], centers=[(lo + hi) / 2])
        zones[z] = {"mae": np.mean(np.abs(pred - x)[m]), "null_mae": np.mean(null[m]),
                    "bias": np.median((pred - x)[m]), "slope": s_z[0],
                    "resolution": np.std((pred - x)[m]) if np.isnan(r_z[0]) else r_z[0]}
    return {"mae": binned_mae(x, np.abs(pred - x)), "null": binned_mae(x, null),
            "slope": slope, "res": res, "zones": zones, "mean": x.mean()}


def zone_resolution(x: np.ndarray, pred: np.ndarray) -> float:
    b, a = np.polyfit(x, pred, 1)
    return np.std(pred - (a + b * x)) / b if b > 0.05 else np.nan


def fig_groups(curves: dict, key: str, ylabel: str, fname: str, ylim, with_null=False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=cm(38, 12), layout="constrained", sharey=True)
    for ax, region in zip(axes, REGION_ORDER, strict=True):
        shade_context(ax)
        for cohort, color, ls in (("naive", REGION_LIGHT[region], "--"),
                                  ("expert", REGION_COLOR[region], "-")):
            sel = [c for (reg, coh, _), c in curves.items() if reg == region and coh == cohort]
            if not sel:
                continue
            stack = np.vstack([c[key] for c in sel])
            for row in stack:
                ax.plot(CENTERS, row, color=color, lw=1, alpha=0.6, ls=ls)
            ax.plot(CENTERS, np.nanmedian(stack, 0), color=color, lw=4, ls=ls,
                    label=f"{cohort} (n={len(sel)})")
            if with_null:
                ax.plot(CENTERS, np.nanmedian(np.vstack([c["null"] for c in sel]), 0),
                        color=color, lw=2, ls=":")
        ax.set_title(REGION_NAME[region], loc="left", fontsize=24, color=REGION_COLOR[region],
                     fontweight="semibold")
        ax.set_xlabel("True position (cm)")
        ax.set_xlim(0, 63)
        ax.set_ylim(*ylim)
        ax.legend(loc="upper right", fontsize=18)
    axes[0].set_ylabel(ylabel)
    fig.savefig(OUT / f"{fname}.svg")
    fig.savefig(OUT / f"{fname}.png", dpi=60)
    plt.close(fig)


def main() -> None:
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_simulation()
    curves, rows = {}, []
    with contextlib.redirect_stdout(io.StringIO()):
        runs = [pdata.load_run(i.stem, "pos") for i in pdata.studies("pos")]
    for r in runs:
        c = recording_curves(r)
        curves[(r.info.region, r.info.cohort, r.info.animal)] = c
        for z, v in c["zones"].items():
            rows.append({"recording": r.info.display, "region": r.info.region,
                         "cohort": r.info.cohort, "zone": z, **v, "train_mean": c["mean"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t1_zone_summary.csv", index=False)
    piv = df.pivot_table(index=["region", "cohort", "recording"], columns="zone",
                         values=["mae", "null_mae", "resolution"]).round(1)
    (OUT / "t1_zone_summary.txt").write_text(piv.to_string())
    fig_groups(curves, "mae", "MAE (cm)  · dotted: no-information decoder", "groups_mae",
               (0, 30), with_null=True)
    fig_groups(curves, "res", "Local resolution (cm)", "groups_resolution", (0, 40))
    fig_groups(curves, "slope", "Local slope (pred vs true)", "groups_slope", (-0.2, 1.2))
    print(piv.to_string())


if __name__ == "__main__":
    main()
