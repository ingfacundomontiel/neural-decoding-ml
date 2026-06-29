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
# # Non-rewarded trial timeout — empirical estimation
#
# In non-rewarded trials (rewCtxt=0 OR rewOdor=0), the rig ends the trial
# by **time**: after the animal exits the corridor, a fixed delay X
# elapses before the teleport. The value of X is not documented; this
# notebook estimates it empirically.
#
# **Procedure**: for each clean non-rewarded trial (no licking), find
# the last bin where position was below the corridor end. The remaining
# bins until `trialFinalBin` are the rig's timer. If X is truly fixed,
# the distribution of those delays should peak sharply at X.
#
# **Sanity check**: rewarded trials where the animal did NOT lick
# (`(1,1,1,0)` cells in our condition cube) should use the *same* timer
# — no licking, no corridor extension. If they peak at the same X,
# the model holds.

# %%
from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent if "__file__" in dir() else Path.cwd().parent
sys.path.insert(0, str(ROOT))

from analysis.loaders import CORRIDOR_END_CM, load_mat, position_in_cm  # noqa: E402

DATA_PATH = Path(os.environ.get(
    "DATASET",
    str(ROOT / "data" / "raw" / "piriform" / "Pir_L5_expert_bins200ms.mat"),
))
OUT_DIR = ROOT / "results" / "exploration" / "01d-non-rewarded-trial-delay" / DATA_PATH.stem
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIN_S = 0.2
THRESHOLD_CM = CORRIDOR_END_CM  # 60 cm

# %% [markdown]
# ## 1. Load and reduce to per-trial labels

# %%
d = load_mat(DATA_PATH)
pos_cm = position_in_cm(d["position"])

trial_final = (d["trialFinalBin"] - 1).astype(int)
trial_start = np.concatenate(([0], trial_final[:-1] + 1))
n_trials = len(trial_final)

rewCtxt_bin = d["rewCtxt"].astype(int)
rewOdor_bin = d["rewOdor"].astype(int)
goResp_bin = d["goResponse"].astype(int)


def per_trial_any(per_bin):
    return np.array([
        int(np.any(per_bin[trial_start[t]: trial_final[t] + 1] == 1))
        for t in range(n_trials)
    ], dtype=int)


rewCtxt_t = per_trial_any(rewCtxt_bin)
rewOdor_t = per_trial_any(rewOdor_bin)
goResp_t = per_trial_any(goResp_bin)

print(f"{n_trials} trials total")

# %% [markdown]
# ## 2. Compute per-trial "delay after last cross of corridor end"
#
# For each trial:
# - Find the last bin where position < `THRESHOLD_CM`.
# - Delay (in seconds) = (trialFinalBin - last_below_bin) × BIN_S.
# - If the animal never crossed THRESHOLD_CM in that trial, the delay is `NaN`.

# %%
def trial_delay(t):
    sl = slice(trial_start[t], trial_final[t] + 1)
    in_trial = pos_cm[sl]
    below = np.where(in_trial < THRESHOLD_CM)[0]
    if len(below) == 0:
        return np.nan
    last_below_local = below[-1]
    last_below_global = trial_start[t] + last_below_local
    return (trial_final[t] - last_below_global) * BIN_S


delays = np.array([trial_delay(t) for t in range(n_trials)])
crossed = ~np.isnan(delays)

print(f"trials that crossed {THRESHOLD_CM:.0f} cm: {int(crossed.sum())} / {n_trials}")

# %% [markdown]
# ## 3. Classify trials and look at the delay per group
#
# - **No-reward, no-lick** `(rewCtxt=0 OR rewOdor=0) AND goResp=0` —
#   the cleanest case for the rig timer.
# - **Reward, no-lick** `(rewCtxt=1 AND rewOdor=1) AND goResp=0` —
#   sanity check: same timer expected.
# - **Reward, lick** `(rewCtxt=1 AND rewOdor=1) AND goResp=1` —
#   control: licking extends the corridor → variable / longer delays.

# %%
no_reward = (rewCtxt_t == 0) | (rewOdor_t == 0)
reward = (rewCtxt_t == 1) & (rewOdor_t == 1)

groups = {
    "no-reward, no-lick": no_reward & (goResp_t == 0) & crossed,
    "reward, no-lick (sanity)": reward & (goResp_t == 0) & crossed,
    "reward, lick (control)": reward & (goResp_t == 1) & crossed,
}

print(f"\n{'group':<28} {'n':>4} {'med':>7} {'p25':>7} {'p75':>7} {'min':>7} {'max':>7}")
print("-" * 70)
for name, mask in groups.items():
    if mask.sum() == 0:
        print(f"{name:<28} {0:>4}    n/a")
        continue
    v = delays[mask]
    print(
        f"{name:<28} {int(mask.sum()):>4} "
        f"{np.median(v):>7.2f} {np.percentile(v, 25):>7.2f} {np.percentile(v, 75):>7.2f} "
        f"{v.min():>7.2f} {v.max():>7.2f}"
    )

# %% [markdown]
# ## 4. Histogram of delays per group

# %%
fig, ax = plt.subplots(figsize=(10, 5))

# Choose bin edges based on the no-reward group (the one we care about most).
no_rew_delays = delays[groups["no-reward, no-lick"]]
if len(no_rew_delays) > 0:
    upper = np.percentile(no_rew_delays, 99) + 2
else:
    upper = 30

bin_edges = np.arange(0, upper + 0.5, 0.5)
colors = {"no-reward, no-lick": "tab:blue", "reward, no-lick (sanity)": "tab:green", "reward, lick (control)": "tab:red"}

for name, mask in groups.items():
    v = delays[mask]
    if len(v) == 0:
        continue
    ax.hist(
        v,
        bins=bin_edges,
        alpha=0.5,
        color=colors[name],
        label=f"{name} (n={int(mask.sum())}, med={np.median(v):.2f} s)",
    )

ax.set_xlabel(f"delay from last bin below {THRESHOLD_CM:.0f} cm to trialFinalBin (s)")
ax.set_ylabel("trials")
ax.set_title(f"Empirical rig timeout estimate — {DATA_PATH.stem}")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "delay-histogram.png", dpi=120)
print(f"saved {OUT_DIR / 'delay-histogram.png'}")

# %% [markdown]
# ## 5. Robustness check — vary the threshold
#
# If the median delay is insensitive to the exact corridor-end threshold,
# we have confidence the geometry assumption is not driving the result.

# %%
print(f"\nMedian delay for no-reward no-lick trials at different thresholds:")
for thr in (50, 55, 60, 65, 70):
    delays_thr = []
    for t in range(n_trials):
        if not (no_reward[t] and goResp_t[t] == 0):
            continue
        sl = slice(trial_start[t], trial_final[t] + 1)
        in_trial = pos_cm[sl]
        below = np.where(in_trial < thr)[0]
        if len(below) == 0:
            continue
        last_below_global = trial_start[t] + below[-1]
        delays_thr.append((trial_final[t] - last_below_global) * BIN_S)
    if delays_thr:
        v = np.array(delays_thr)
        print(f"  threshold={thr:>3} cm:  n={len(v):>3}  median={np.median(v):>5.2f} s  p25={np.percentile(v, 25):>5.2f}  p75={np.percentile(v, 75):>5.2f}")
    else:
        print(f"  threshold={thr:>3} cm:  n=0")

plt.close("all")
