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
# # Position diagnostic — raw values, no cm conversion
#
# Companion to `01-trial-structure.py`. Re-plots position-related figures
# **using the raw values straight from the `.mat`**, without applying any
# `cm/degree` or `mm → cm` conversion. Purpose: see what units the `position`
# variable actually lives in, so the right conversion factor can be confirmed
# with the director.
#
# Outputs live under `results/exploration/01b-position-raw-diagnostic/<dataset>/`
# to avoid clobbering the main notebook's figures.

# %%
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from analysis.loaders import load_mat  # noqa: E402

DATA_PATH = Path(os.environ.get(
    "DATASET",
    str(ROOT / "data" / "raw" / "piriform" / "Pir_L5_expert_bins200ms.mat"),
))
OUT_DIR = ROOT / "results" / "exploration" / "01b-position-raw-diagnostic" / DATA_PATH.stem
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_S = 0.2

# %% [markdown]
# ## 1. Raw stats

# %%
d = load_mat(DATA_PATH)
pos = d["position"]

print(f"dataset: {DATA_PATH.stem}")
print(f"position dtype : {pos.dtype}")
print(f"position shape : {pos.shape}")
print()
print(f"  min   : {pos.min():>10.2f}")
print(f"  p01   : {np.percentile(pos, 1):>10.2f}")
print(f"  p25   : {np.percentile(pos, 25):>10.2f}")
print(f"  median: {np.median(pos):>10.2f}")
print(f"  mean  : {pos.mean():>10.2f}")
print(f"  p75   : {np.percentile(pos, 75):>10.2f}")
print(f"  p99   : {np.percentile(pos, 99):>10.2f}")
print(f"  max   : {pos.max():>10.2f}")
print()
print("Hypothetical conversion to cm under different assumptions:")
print(f"  if raw is mm        : max in cm = {pos.max() / 10:.2f}")
print(f"  if raw is degrees w/ 0.1222 factor : max in cm = {pos.max() * 0.1222:.2f}")
print(f"  declared corridor   : ~60-65 cm")

# %% [markdown]
# ## 2. Trial-level aggregates

# %%
trial_final = (d["trialFinalBin"] - 1).astype(int)
trial_start = np.concatenate(([0], trial_final[:-1] + 1))
rewCtxt_bin = d["rewCtxt"].astype(int)
rewOdor_bin = d["rewOdor"].astype(int)
goResp_bin = d["goResponse"].astype(int)
n_trials = len(trial_final)

rewCtxt_trial = np.empty(n_trials, dtype=int)
rewOdor_trial = np.empty(n_trials, dtype=int)
goResp_trial = np.empty(n_trials, dtype=int)
for t in range(n_trials):
    sl = slice(trial_start[t], trial_final[t] + 1)
    rewCtxt_trial[t] = int(np.any(rewCtxt_bin[sl] == 1))
    rewOdor_trial[t] = int(np.any(rewOdor_bin[sl] == 1))
    goResp_trial[t] = int(np.any(goResp_bin[sl] == 1))

max_pos_raw = np.array([pos[trial_start[t]: trial_final[t] + 1].max() for t in range(n_trials)])

print(f"{'rewC':>4} {'rewO':>4} {'goR':>4}  {'n':>4}  {'max_med_raw':>12}  {'max_p90_raw':>12}")
print("-" * 55)
for rc in (0, 1):
    for ro in (0, 1):
        for gr in (0, 1):
            mask = (rewCtxt_trial == rc) & (rewOdor_trial == ro) & (goResp_trial == gr)
            n = int(mask.sum())
            if n == 0:
                continue
            med = np.median(max_pos_raw[mask])
            p90 = np.percentile(max_pos_raw[mask], 90)
            tag = "  ← drift candidate" if (rc, ro, gr) == (1, 1, 1) else ""
            print(f"{rc:>4} {ro:>4} {gr:>4}  {n:>4}  {med:>12.2f}  {p90:>12.2f}{tag}")

# %% [markdown]
# ## 3. Histogram of all raw positions

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(pos, bins=120, alpha=0.75, color="steelblue", edgecolor="white")
ax.set_xlabel("position (raw units, as stored in .mat)")
ax.set_ylabel("bin count (whole recording)")
ax.set_title(f"Raw position distribution — {DATA_PATH.stem}")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "histogram-raw.png", dpi=120)
print(f"saved {OUT_DIR / 'histogram-raw.png'}")

# %% [markdown]
# ## 4. Boxplot of per-trial max position (raw) by condition

# %%
group_labels = []
group_values = []
for rc in (0, 1):
    for ro in (0, 1):
        for gr in (0, 1):
            mask = (rewCtxt_trial == rc) & (rewOdor_trial == ro) & (goResp_trial == gr)
            if mask.sum() == 0:
                continue
            group_labels.append(f"({rc},{ro},{gr})\nn={int(mask.sum())}")
            group_values.append(max_pos_raw[mask])

fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot(group_values, tick_labels=group_labels, showfliers=False)
ax.set_xlabel("(rewCtxt, rewOdor, goResp)")
ax.set_ylabel("max position per trial (raw)")
ax.set_title(f"Per-trial max position by condition — {DATA_PATH.stem} (raw units)")
ax.tick_params(axis="x", labelsize=8)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "extrema-raw.png", dpi=120)
print(f"saved {OUT_DIR / 'extrema-raw.png'}")

# %% [markdown]
# ## 5. Sample trajectories by condition (raw position vs time)

# %%
rng = np.random.default_rng(42)
sample_size = 5

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
fig.suptitle(f"Position within trial (raw units) — {DATA_PATH.stem}")

for i, rc in enumerate((0, 1)):
    for j, ro in enumerate((0, 1)):
        ax = axes[i, j]
        for gr, color, label in [(0, "tab:gray", "goResp=0"), (1, "tab:red", "goResp=1")]:
            mask = (rewCtxt_trial == rc) & (rewOdor_trial == ro) & (goResp_trial == gr)
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            picks = rng.choice(idx, size=min(sample_size, len(idx)), replace=False)
            for p in picks:
                t_bins = np.arange(trial_start[p], trial_final[p] + 1)
                t_s = (t_bins - trial_start[p]) * BIN_S
                ax.plot(t_s, pos[t_bins], color=color, alpha=0.6, lw=1)
            ax.plot([], [], color=color, label=f"{label} (of {len(idx)})")
        ax.set_title(f"rewCtxt={rc}, rewOdor={ro}")
        ax.set_xlabel("time within trial (s)")
        ax.set_ylabel("position (raw)")
        ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT_DIR / "trajectories-raw.png", dpi=120)
print(f"saved {OUT_DIR / 'trajectories-raw.png'}")

plt.close("all")
