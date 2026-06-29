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
# # Trial structure exploration
#
# First-pass exploration on a single sample dataset.
#
# **Question:** do trials with `(rewCtxt=1, rewOdor=1, goResponse=1)` show
# longer durations and a position drift consistent with the virtual corridor
# extending while the animal licks at the reward?
#
# Sample dataset: `Pir_L5_expert_bins200ms.mat`.
#
# **Note on dimensionality** (learned the hard way on first run): the `.mat`
# stores variables at two levels.
# - **Per-bin** (`T_bins`): `neuronActivity`, `position`, `velocity`,
#   `rewCtxt`, `rewOdor`, `goResponse` — they hold the value at each time
#   bin.
# - **Per-trial** (`E_trials`): `trialDurationInBins`, `trialFinalBin`,
#   `dPrime`, `criterion` — one value per trial.
#
# To do trial-level analysis with the per-bin labels, we either pick a
# representative bin per trial (for `rewCtxt`, `rewOdor` — assumed constant
# within a trial) or aggregate with `np.any` (for `goResponse` — dynamic).

# %%
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from analysis.loaders import (  # noqa: E402
    CM_PER_DEGREE,
    CONTEXT_END_CM,
    CONTEXT_START_CM,
    CORRIDOR_END_CM,
    check_expected,
    load_mat,
    position_in_cm,
)

import os

# DATASET env var lets us point the same script at any .mat without editing.
# Default = Pir_L5_expert_bins200ms (sample dataset for first-pass exploration).
DATA_PATH = Path(os.environ.get(
    "DATASET",
    str(ROOT / "data" / "raw" / "piriform" / "Pir_L5_expert_bins200ms.mat"),
))
OUT_DIR = ROOT / "results" / "exploration" / "01-trial-structure" / DATA_PATH.stem
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_S = 0.2  # 200 ms bins (200 ms files); adjust if a 100 ms dataset is passed.

# %% [markdown]
# ## 1. Load and inspect

# %%
d = load_mat(DATA_PATH)

print("Variables present:", sorted(d.keys()))
print()
for k in sorted(d.keys()):
    v = d[k]
    shape = getattr(v, "shape", "()")
    dtype = getattr(v, "dtype", type(v).__name__)
    print(f"  {k:24s}  shape={str(shape):>20s}  dtype={dtype}")

print()
print("Expected-vars check:", check_expected(d))

# %% [markdown]
# ## 2. Position units sanity check

# %%
pos = d["position"]
pos_cm = position_in_cm(pos)
print(f"position (raw deg) : min={pos.min():8.2f}   max={pos.max():8.2f}   mean={pos.mean():8.2f}")
print(f"position (cm)      : min={pos_cm.min():8.2f}   max={pos_cm.max():8.2f}   mean={pos_cm.mean():8.2f}")
print()
print(f"Natural corridor end = {CORRIDOR_END_CM} cm. Anything above that is drift.")
print(f"Context: {CONTEXT_START_CM} → {CONTEXT_END_CM} cm.")
print(f"Observed max here: {pos_cm.max():.1f} cm (excess over corridor: {pos_cm.max() - CORRIDOR_END_CM:+.1f} cm).")

# %% [markdown]
# ## 3. Aggregate per-bin labels into per-trial labels

# %%
trial_final = (d["trialFinalBin"] - 1).astype(int)  # MATLAB 1-indexed → Python 0-indexed
trial_start = np.concatenate(([0], trial_final[:-1] + 1))
n_trials = len(trial_final)
n_bins = len(pos)

rewCtxt_bin = d["rewCtxt"].astype(int)
rewOdor_bin = d["rewOdor"].astype(int)
goResp_bin = d["goResponse"].astype(int)

# IMPORTANT — rewCtxt / rewOdor are NOT constant within each trial: they look
# like presentation flags (1 only during the rewarded-context / odor-on window,
# not for the whole trial). So we define the trial-level label as "did this
# condition occur at any point during the trial?" — np.any over the trial's
# bins. Same logic for goResponse (animal licked at any time).
rewCtxt_trial = np.zeros(n_trials, dtype=int)
rewOdor_trial = np.zeros(n_trials, dtype=int)
goResp_trial = np.zeros(n_trials, dtype=int)
# Also keep a coverage metric: what fraction of bins in the trial had the flag on.
rewCtxt_cov = np.zeros(n_trials)
rewOdor_cov = np.zeros(n_trials)
for t in range(n_trials):
    sl = slice(trial_start[t], trial_final[t] + 1)
    rewCtxt_trial[t] = int(np.any(rewCtxt_bin[sl] == 1))
    rewOdor_trial[t] = int(np.any(rewOdor_bin[sl] == 1))
    goResp_trial[t] = int(np.any(goResp_bin[sl] == 1))
    L = sl.stop - sl.start
    rewCtxt_cov[t] = float((rewCtxt_bin[sl] == 1).sum()) / L
    rewOdor_cov[t] = float((rewOdor_bin[sl] == 1).sum()) / L

print(f"Total bins: {n_bins}    Total trials: {n_trials}")
print()
print("Fraction of trial bins where the flag is on (median across trials with flag=1):")
for name, cov, lab in [("rewCtxt", rewCtxt_cov, rewCtxt_trial),
                       ("rewOdor", rewOdor_cov, rewOdor_trial)]:
    on_trials = cov[lab == 1]
    if len(on_trials):
        print(f"  {name:>8} : median coverage = {np.median(on_trials):.2f}   "
              f"(over {len(on_trials)} trials where flag fired at all)")

# %% [markdown]
# ## 4. Trial composition by reward condition (per-trial counts)

# %%
print(f"{'rewCtxt':>8} {'rewOdor':>8} {'goResp':>8} {'trials':>7} {'percent':>9}")
print("-" * 46)
for rc in (0, 1):
    for ro in (0, 1):
        for gr in (0, 1):
            mask = (rewCtxt_trial == rc) & (rewOdor_trial == ro) & (goResp_trial == gr)
            n = int(mask.sum())
            pct = 100 * n / n_trials
            marker = "  ← drift candidate" if (rc, ro, gr) == (1, 1, 1) else ""
            print(f"{rc:>8} {ro:>8} {gr:>8} {n:>7} {pct:>8.1f}%{marker}")

# %% [markdown]
# ## 5. V1 — Trial duration distributions by condition

# %%
trial_dur = d["trialDurationInBins"].astype(int)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
fig.suptitle("Trial duration (bins) by (rewCtxt, rewOdor) — split by goResponse")

dmax = int(trial_dur.max())
bins_edges = np.arange(0, dmax + 6, 5)

for i, rc in enumerate((0, 1)):
    for j, ro in enumerate((0, 1)):
        ax = axes[i, j]
        for gr, color, label in [(0, "tab:gray", "goResp=0"), (1, "tab:red", "goResp=1")]:
            mask = (rewCtxt_trial == rc) & (rewOdor_trial == ro) & (goResp_trial == gr)
            if mask.sum() > 0:
                ax.hist(
                    trial_dur[mask],
                    bins=bins_edges,
                    alpha=0.6,
                    color=color,
                    label=f"{label}  n={int(mask.sum())}  median={int(np.median(trial_dur[mask]))}",
                )
        ax.set_title(f"rewCtxt={rc}, rewOdor={ro}")
        ax.set_xlabel("trial duration (bins)")
        ax.set_ylabel("count")
        ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT_DIR / "v1-duration-histograms.png", dpi=120)
print(f"saved {OUT_DIR / 'v1-duration-histograms.png'}")

# %% [markdown]
# ## 6. V2 — Position trajectories of sample trials

# %%
rng = np.random.default_rng(42)
sample_size = 6

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
fig.suptitle("Position within trial (cm) — sample trials by condition")

for i, rc in enumerate((0, 1)):
    for j, ro in enumerate((0, 1)):
        ax = axes[i, j]
        # Corridor geometry: context band + natural end line.
        ax.axhspan(CONTEXT_START_CM, CONTEXT_END_CM, color="gold", alpha=0.18,
                   label="context (42-59 cm)" if (i, j) == (0, 0) else None)
        ax.axhline(CORRIDOR_END_CM, color="k", linestyle="--", alpha=0.4, lw=0.8,
                   label=f"corridor end ({CORRIDOR_END_CM} cm)" if (i, j) == (0, 0) else None)
        for gr, color, label in [(0, "tab:gray", "goResp=0"), (1, "tab:red", "goResp=1")]:
            mask = (rewCtxt_trial == rc) & (rewOdor_trial == ro) & (goResp_trial == gr)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            picks = rng.choice(idx, size=min(sample_size, len(idx)), replace=False)
            for p in picks:
                t_bins = np.arange(trial_start[p], trial_final[p] + 1)
                t_s = (t_bins - trial_start[p]) * BIN_S
                ax.plot(t_s, position_in_cm(pos[t_bins]), color=color, alpha=0.6, lw=1)
            ax.plot([], [], color=color, label=f"{label} (n={len(idx)} total)")
        ax.set_title(f"rewCtxt={rc}, rewOdor={ro}")
        ax.set_xlabel("time within trial (s)")
        ax.set_ylabel("position (cm)")
        ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT_DIR / "v2-position-trajectories.png", dpi=120)
print(f"saved {OUT_DIR / 'v2-position-trajectories.png'}")

# %% [markdown]
# ## 7. V3 — Per-trial position extrema

# %%
max_pos = np.array([
    position_in_cm(pos[trial_start[t]: trial_final[t] + 1]).max()
    for t in range(n_trials)
])
final_pos = np.array([position_in_cm(pos[trial_final[t]]) for t in range(n_trials)])

group_labels = []
vals_max = []
vals_final = []
for rc in (0, 1):
    for ro in (0, 1):
        for gr in (0, 1):
            mask = (rewCtxt_trial == rc) & (rewOdor_trial == ro) & (goResp_trial == gr)
            if mask.sum() == 0:
                continue
            group_labels.append(f"({rc},{ro},{gr})\nn={int(mask.sum())}")
            vals_max.append(max_pos[mask])
            vals_final.append(final_pos[mask])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, vals, title in zip(
    axes,
    [vals_max, vals_final],
    ["max position per trial (cm)", "final-bin position per trial (cm)"],
):
    ax.boxplot(vals, tick_labels=group_labels, showfliers=False)
    ax.axhspan(CONTEXT_START_CM, CONTEXT_END_CM, color="gold", alpha=0.18, label="context zone")
    ax.axhline(CORRIDOR_END_CM, color="k", linestyle="--", alpha=0.4, lw=0.8,
               label=f"corridor end ({CORRIDOR_END_CM} cm)")
    ax.set_title(title)
    ax.set_xlabel("(rewCtxt, rewOdor, goResp)")
    ax.set_ylabel("cm")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

fig.suptitle("Per-trial position extrema by condition")
fig.tight_layout()
fig.savefig(OUT_DIR / "v3-position-extrema.png", dpi=120)
print(f"saved {OUT_DIR / 'v3-position-extrema.png'}")

# %% [markdown]
# ## 8. Numeric summary by condition

# %%
print(f"{'rewC':>4} {'rewO':>4} {'goR':>4} {'n_trials':>9} "
      f"{'dur_med':>8} {'dur_p90':>8} {'max_med':>8} {'max_p90':>8}")
print("-" * 64)
for rc in (0, 1):
    for ro in (0, 1):
        for gr in (0, 1):
            mask = (rewCtxt_trial == rc) & (rewOdor_trial == ro) & (goResp_trial == gr)
            if mask.sum() == 0:
                continue
            dur = trial_dur[mask]
            mx = max_pos[mask]
            print(
                f"{rc:>4} {ro:>4} {gr:>4} {int(mask.sum()):>9} "
                f"{int(np.median(dur)):>8} {int(np.percentile(dur, 90)):>8} "
                f"{np.median(mx):>8.1f} {np.percentile(mx, 90):>8.1f}"
            )

# %% [markdown]
# ## 9. V4 — Velocity check: lickers vs runners in drift territory
#
# For trials where the animal goes past the corridor end, what does velocity
# look like? Two hypotheses, two distinct signatures:
#
# - **Lickers** `(1,1,1)`: position should plateau near ~80 cm while velocity
#   collapses to ~0 (animal stationary at reward zone, the rig extends the
#   virtual corridor on its own).
# - **Runners** `(0,*,*)`: position should keep climbing past 60 cm while
#   velocity stays positive (animal actively running through the extended
#   corridor because nothing tells it to stop).

# %%
vel = d.get("velocity", d.get("speed"))
if vel is None:
    print("No velocity/speed variable in this dataset — skipping V4.")
else:
    def longest_trials(rc, ro, gr, n=3):
        mask = (rewCtxt_trial == rc) & (rewOdor_trial == ro) & (goResp_trial == gr)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        return sorted(idx, key=lambda t: trial_dur[t], reverse=True)[:n]

    conditions_to_show = [
        ("(1,1,1) lickers", longest_trials(1, 1, 1, 3), "tab:red"),
        ("(0,1,0) runners", longest_trials(0, 1, 0, 3), "tab:blue"),
        ("(0,0,0) baseline", longest_trials(0, 0, 0, 3), "tab:gray"),
    ]

    # V4a — position + velocity overlay for the longest trials of each condition
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    for ax_pos, (cond_label, trial_ids, color) in zip(axes, conditions_to_show):
        ax_vel = ax_pos.twinx()
        for t in trial_ids:
            t_bins = np.arange(trial_start[t], trial_final[t] + 1)
            t_s = (t_bins - trial_start[t]) * BIN_S
            ax_pos.plot(t_s, position_in_cm(pos[t_bins]), color=color, alpha=0.9, lw=1.4)
            ax_vel.plot(t_s, vel[t_bins], color="black", alpha=0.45, lw=0.8, linestyle="--")
        ax_pos.axhline(CORRIDOR_END_CM, color="k", linestyle=":", alpha=0.4)
        ax_pos.set_ylabel("position (cm)", color=color)
        ax_vel.set_ylabel("velocity (raw units)", color="black")
        ax_pos.set_title(f"{cond_label} — longest {len(trial_ids)} trials  "
                         f"(position solid, velocity dashed)")
        ax_pos.set_xlabel("time within trial (s)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "v4a-velocity-overlay.png", dpi=120)
    print(f"saved {OUT_DIR / 'v4a-velocity-overlay.png'}")

    # V4b — Scatter of (position, velocity) for bins past the corridor end
    drift_bins = pos_cm > CORRIDOR_END_CM

    # Per-bin condition label, derived from trial-level labels
    cond_per_bin = np.full(n_bins, "other", dtype=object)
    for t in range(n_trials):
        sl = slice(trial_start[t], trial_final[t] + 1)
        if rewCtxt_trial[t] == 1 and rewOdor_trial[t] == 1 and goResp_trial[t] == 1:
            cond_per_bin[sl] = "(1,1,1) lick"
        elif rewCtxt_trial[t] == 0:
            cond_per_bin[sl] = "(0,*,*) no-reward"

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, color in [
        ("(0,*,*) no-reward", "tab:blue"),
        ("(1,1,1) lick", "tab:red"),
        ("other", "lightgray"),
    ]:
        mask = drift_bins & (cond_per_bin == label)
        if mask.sum() > 0:
            ax.scatter(pos_cm[mask], vel[mask], color=color, alpha=0.35, s=6,
                       label=f"{label}  (n bins = {int(mask.sum())})")
    ax.axvline(CORRIDOR_END_CM, color="k", linestyle=":", alpha=0.5)
    ax.axhline(0, color="k", linestyle="-", alpha=0.3, lw=0.5)
    ax.set_xlabel("position (cm)")
    ax.set_ylabel("velocity (raw units)")
    ax.set_title("Velocity vs position — only bins past the corridor end")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "v4b-velocity-vs-position.png", dpi=120)
    print(f"saved {OUT_DIR / 'v4b-velocity-vs-position.png'}")

# %% [markdown]
# ## 10. V5 — Empirical position distribution
#
# Histogram of every position sample (across the whole recording, all bins).
# Used to spot the true empirical corridor end: if there's a clear inflection
# or fall-off at some position, that's likely where the visual corridor
# physically ends. The 60 cm constant is the lab's stated value — empirical
# data may suggest tightening or loosening it.

# %%
fig, ax = plt.subplots(figsize=(11, 4))
ax.hist(pos_cm, bins=120, color="steelblue", alpha=0.8, edgecolor="white")
ax.axvspan(CONTEXT_START_CM, CONTEXT_END_CM, color="gold", alpha=0.25, label="context zone")
ax.axvline(CORRIDOR_END_CM, color="k", linestyle="--", alpha=0.7,
           label=f"declared corridor end ({CORRIDOR_END_CM} cm)")
ax.set_xlabel("position (cm)")
ax.set_ylabel("bin count (whole recording)")
ax.set_title(f"Empirical position distribution — {DATA_PATH.stem}")
ax.legend(loc="best")
fig.tight_layout()
fig.savefig(OUT_DIR / "v5-position-histogram.png", dpi=120)
print(f"saved {OUT_DIR / 'v5-position-histogram.png'}")

plt.close("all")
