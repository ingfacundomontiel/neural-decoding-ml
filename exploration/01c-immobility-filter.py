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
# # Immobility filter — apply Sebas's `periods_immobility` to the sample dataset
#
# Companion to `01-trial-structure.py`. Uses the borrowed `periods_immobility`
# from `analysis/preprocessing.py` (adapted from Sebas's `feat/pre-processing`
# branch) to identify bins that the lab's pipeline would remove from
# decoding, and checks whether those bins coincide with the lick-drift
# cluster we identified visually in V4b.
#
# Thresholds used (taken directly from
# `external/scriptspipeline/all-datasets-experiment.py`):
#
# - `thVel = 1` cm/s
# - `thDur = 4` seconds
# - `binLength = 200` ms
# - `bins_before = bins_after = 5` (matches Sebas's typical history window)

# %%
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from analysis.loaders import CM_PER_DEGREE, load_mat, position_in_cm  # noqa: E402
from ibioml.preprocess_data import periods_immobility  # noqa: E402

DATA_PATH = Path(os.environ.get(
    "DATASET",
    str(ROOT / "data" / "raw" / "piriform" / "Pir_L5_expert_bins200ms.mat"),
))
OUT_DIR = ROOT / "results" / "exploration" / "01c-immobility-filter" / DATA_PATH.stem
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_MS = 200       # bin length in milliseconds
BIN_S = BIN_MS / 1000
THVEL = 1          # cm/s (Sebas)
THDUR = 4          # seconds (Sebas)
BINS_BEFORE = 5
BINS_AFTER = 5

CORRIDOR_END_CM_DECLARED = 60.0  # for visual reference only

# %% [markdown]
# ## 1. Load + per-trial aggregates (same recipe as 01)

# %%
d = load_mat(DATA_PATH)
pos = d["position"]
vel = d["velocity"]
pos_cm = position_in_cm(pos)
n_bins = len(pos)

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

print(f"dataset: {DATA_PATH.stem}")
print(f"  bins  : {n_bins}")
print(f"  trials: {n_trials}")

# %% [markdown]
# ## 2. Apply Sebas's `periods_immobility`

# %%
immobility, trial_transitions = periods_immobility(
    vels_binned=vel,
    thVel=THVEL,
    binLength=BIN_MS,
    thDur=THDUR,
    trialFinalBin=trial_final,
    bins_before=BINS_BEFORE,
    bins_after=BINS_AFTER,
)

immob_set = set(immobility)
trans_set = set(trial_transitions)
both = immob_set & trans_set
either = immob_set | trans_set

immob_mask = np.zeros(n_bins, dtype=bool)
immob_mask[np.array(sorted(b for b in immob_set if 0 <= b < n_bins), dtype=int)] = True
trans_mask = np.zeros(n_bins, dtype=bool)
trans_mask[np.array(sorted(b for b in trans_set if 0 <= b < n_bins), dtype=int)] = True
either_mask = immob_mask | trans_mask

print(f"\nImmobility bins   : {len(immob_set):>6} ({100 * immob_mask.sum() / n_bins:.1f}% of all bins)")
print(f"Trial-transition  : {len(trans_set):>6} ({100 * trans_mask.sum() / n_bins:.1f}% of all bins)")
print(f"Overlap           : {len(both):>6}")
print(f"Union (any filter): {len(either):>6} ({100 * either_mask.sum() / n_bins:.1f}% of all bins)")
print(f"\nThresholds used: thVel={THVEL} cm/s, thDur={THDUR} s, bins ±{BINS_BEFORE}/{BINS_AFTER}")

# %% [markdown]
# ## 3. F1 — Velocity distribution with the `thVel` marker

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(vel, bins=100, alpha=0.75, color="steelblue", edgecolor="white")
ax.axvline(THVEL, color="red", linestyle="--", label=f"thVel = {THVEL} cm/s")
ax.set_xlabel("velocity (cm/s)")
ax.set_ylabel("bin count")
ax.set_title(f"Velocity distribution — {DATA_PATH.stem}")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "f1-velocity-histogram.png", dpi=120)
print(f"saved {OUT_DIR / 'f1-velocity-histogram.png'}")

# %% [markdown]
# ## 4. F2 — Duration of low-velocity stretches, with `thDur` marker
#
# (Computed inside `periods_immobility` and exposed indirectly. Re-derive
# here for plotting.)

# %%
candidates = (np.ravel(vel) < THVEL).astype(int)
candidates[-1] = 0
jumps = np.append(candidates[0], np.diff(candidates))
ev_on = np.where(jumps == 1)[0]
ev_off = np.where(jumps == -1)[0]
durs_s = (ev_off - ev_on) * BIN_S

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(durs_s, bins=50, alpha=0.75, color="steelblue", edgecolor="white")
ax.axvline(THDUR, color="red", linestyle="--", label=f"thDur = {THDUR} s")
ax.set_xlabel("low-velocity stretch duration (s)")
ax.set_ylabel("count of stretches")
ax.set_title(f"Low-velocity stretch durations — {len(durs_s)} stretches total")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "f2-stretch-durations.png", dpi=120)
print(f"saved {OUT_DIR / 'f2-stretch-durations.png'}")
print(f"  stretches above thDur: {(durs_s > THDUR).sum()} of {len(durs_s)}")

# %% [markdown]
# ## 5. F3 — Overlay of filtered bins on the V4b scatter
#
# For bins past the declared corridor end, colour by:
# - filtered as immobility,
# - filtered as trial-transition,
# - kept.
#
# Expectation: the immobility-filtered bins should pile up near the bottom
# of the velocity axis (≈ 0), exactly where the lick cluster from V4b sits.

# %%
drift_mask = pos_cm > CORRIDOR_END_CM_DECLARED

fig, ax = plt.subplots(figsize=(10, 6))
keep_mask = drift_mask & ~either_mask
imm_only = drift_mask & immob_mask & ~trans_mask
trans_only = drift_mask & trans_mask & ~immob_mask
both_filt = drift_mask & immob_mask & trans_mask

ax.scatter(pos_cm[keep_mask], vel[keep_mask], color="lightgray", alpha=0.4, s=4,
           label=f"kept (n={keep_mask.sum()})")
ax.scatter(pos_cm[imm_only], vel[imm_only], color="tab:red", alpha=0.5, s=4,
           label=f"immobility only (n={imm_only.sum()})")
ax.scatter(pos_cm[trans_only], vel[trans_only], color="tab:blue", alpha=0.5, s=4,
           label=f"trial-transition only (n={trans_only.sum()})")
ax.scatter(pos_cm[both_filt], vel[both_filt], color="purple", alpha=0.5, s=4,
           label=f"both (n={both_filt.sum()})")

ax.axhline(THVEL, color="red", linestyle=":", alpha=0.6, lw=1)
ax.axvline(CORRIDOR_END_CM_DECLARED, color="k", linestyle=":", alpha=0.5)
ax.axhline(0, color="k", linestyle="-", alpha=0.3, lw=0.5)
ax.set_xlabel(f"position (cm, using CM_PER_DEGREE={CM_PER_DEGREE})")
ax.set_ylabel("velocity (cm/s)")
ax.set_title("Filter coverage on the velocity-position drift scatter")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "f3-filter-on-v4b.png", dpi=120)
print(f"saved {OUT_DIR / 'f3-filter-on-v4b.png'}")

# %% [markdown]
# ## 6. F4 — Per-condition fraction of bins filtered

# %%
group_labels = []
frac_immob = []
frac_trans = []
frac_either = []

for rc in (0, 1):
    for ro in (0, 1):
        for gr in (0, 1):
            mask = (rewCtxt_trial == rc) & (rewOdor_trial == ro) & (goResp_trial == gr)
            if mask.sum() == 0:
                continue
            # bins belonging to these trials
            trial_bins_mask = np.zeros(n_bins, dtype=bool)
            for t in np.where(mask)[0]:
                trial_bins_mask[trial_start[t]: trial_final[t] + 1] = True
            n_in = trial_bins_mask.sum()
            group_labels.append(f"({rc},{ro},{gr})\nn_bins={n_in}")
            frac_immob.append(100 * (trial_bins_mask & immob_mask).sum() / n_in)
            frac_trans.append(100 * (trial_bins_mask & trans_mask).sum() / n_in)
            frac_either.append(100 * (trial_bins_mask & either_mask).sum() / n_in)

x = np.arange(len(group_labels))
w = 0.27

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w, frac_immob, w, label="immobility", color="tab:red", alpha=0.8)
ax.bar(x, frac_trans, w, label="trial-transition", color="tab:blue", alpha=0.8)
ax.bar(x + w, frac_either, w, label="either (union)", color="purple", alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=8)
ax.set_xlabel("(rewCtxt, rewOdor, goResp)")
ax.set_ylabel("% of the condition's bins filtered")
ax.set_title("Filter coverage by condition")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "f4-fraction-by-condition.png", dpi=120)
print(f"saved {OUT_DIR / 'f4-fraction-by-condition.png'}")

plt.close("all")
