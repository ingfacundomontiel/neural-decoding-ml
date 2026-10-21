"""Sample data + preprocessing in one figure (test version, plan.md section 4).

Rows share the real time axis of a few consecutive trials of one recording:
1. raw spike counts of every recorded neuron (neurons dropped by quality control
   drawn in grey);
2. the decoder input — Gaussian-smoothed counts (σ = 200 ms) of the kept
   neurons, removed bins masked;
3. position, removed bins shaded by cause, context zone marked;
4. speed, with the immobility threshold.
Removal logic is the library's own (ibioml.preprocess_data), final pipeline.
"""
from __future__ import annotations

import contextlib
import io
import os

import numpy as np
from matplotlib.patches import Patch

import poster_data as pdata
from poster_style import (CONTEXT_EXIT_CM, INK, INK_SOFT, REGION_COLOR,
                          REMOVED_COLOR, cm, save, setup)
from analysis.loaders import CM_PER_DEGREE
from ibioml.preprocess_data import (clean_neurons_by_low_firing_rate, clean_unstable_neurons,
                                    get_idx_by_position_tail, periods_immobility,
                                    smooth_neural_data)

import matplotlib.pyplot as plt

STEM = os.environ.get("STEM", "Pir_N31_expert_bins200ms")
TRIALS = (int(os.environ.get("T0", "97")), int(os.environ.get("T1", "100")))
SIZE = (38.0, 24.5)
BIN_S = 0.2
CAUSE_LABEL = {"transition": "teleport transition", "immobility": "immobility (> 4 s under 1 cm/s)",
               "tail": "past the context exit (> 62.9 cm)"}


def removal_causes(rec: dict, n: int) -> np.ndarray:
    """Per raw bin: '' (kept) or the cause shown for it (tail > immobility > transition)."""
    tfb = rec["trialFinalBin"].astype(int) - 1
    with contextlib.redirect_stdout(io.StringIO()):
        immob, trans = periods_immobility(rec["velocity"], 1, 200, 4, tfb, 0, 0)
        tail = get_idx_by_position_tail(rec["position"], tfb, pdata.TAIL_DEG, 0)
    cause = np.full(n, "", dtype=object)
    cause[np.array(trans, dtype=int)] = "transition"
    cause[np.array(immob, dtype=int)] = "immobility"
    cause[np.asarray(tail, dtype=int)] = "tail"
    return cause


def kept_neurons(rec: dict, kept_bins: np.ndarray) -> np.ndarray:
    smoothed = smooth_neural_data(rec["spikes"], 200)[kept_bins]
    drop = np.union1d(clean_unstable_neurons(smoothed),
                      clean_neurons_by_low_firing_rate(smoothed, 0.1, 200)[0]).astype(int)
    return np.setdiff1d(np.arange(rec["spikes"].shape[1]), drop)


def raster(ax, counts: np.ndarray, t0: float, keep_mask: np.ndarray) -> None:
    order = np.argsort(counts.sum(axis=0))  # least active at the bottom
    for row, j in enumerate(order):
        b = np.nonzero(counts[:, j])[0]
        times = np.concatenate([t0 + (i + (np.arange(int(counts[i, j])) + 0.5) / counts[i, j]) * BIN_S
                                for i in b]) if len(b) else np.array([])
        ax.vlines(times, row + 0.1, row + 0.9, color=INK if keep_mask[j] else "#A8A8A8", lw=0.9)
    ax.set_ylim(0, counts.shape[1])
    ax.set_yticks([0.5, counts.shape[1] - 0.5], ["1", str(counts.shape[1])])
    ax.set_ylabel("Neuron")


def main() -> None:
    setup()
    info = next(i for i in pdata.studies("pos") if i.stem == STEM)
    rec = pdata.raw_recording(info.stem, info.region, info.cohort)
    n = len(rec["position"])
    with contextlib.redirect_stdout(io.StringIO()):
        kept = pdata.kept_raw_indices(rec, info.cohort)
    neurons = kept_neurons(rec, kept)
    keep_mask = np.isin(np.arange(rec["spikes"].shape[1]), neurons)
    tfb = rec["trialFinalBin"].astype(int) - 1
    a, b = tfb[TRIALS[0] - 1] + 1, tfb[TRIALS[1]] + 1
    t = np.arange(a, b) * BIN_S
    t -= t[0]
    cause = removal_causes(rec, n)[a:b]
    is_kept = np.isin(np.arange(a, b), kept)

    fig, axes = plt.subplots(4, 1, figsize=cm(*SIZE), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [2.6, 2.4, 2.6, 2.0]})
    ax_r, ax_h, ax_p, ax_v = axes
    raster(ax_r, rec["spikes"][a:b], 0.0, keep_mask)
    ax_r.set_title(f"{info.animal} · raw spike counts, {rec['spikes'].shape[1]} neurons "
                   f"(grey: removed by quality control)", loc="left", fontsize=24)

    smoothed = smooth_neural_data(rec["spikes"], 200)[a:b][:, neurons]
    z = (smoothed - smoothed.mean(0)) / (smoothed.std(0) + 1e-9)
    z[~is_kept] = np.nan
    order = np.argsort(rec["spikes"][:, neurons].sum(axis=0))
    ax_h.imshow(z[:, order].T, aspect="auto", origin="lower", cmap="Greys", vmin=-1, vmax=3,
                extent=(0, t[-1] + BIN_S, 0, len(neurons)), interpolation="nearest")
    ax_h.set_yticks([0.5, len(neurons) - 0.5], ["1", str(len(neurons))])
    ax_h.set_ylabel("Neuron")
    ax_h.set_title(f"decoder input: smoothed (σ = 200 ms), {len(neurons)} kept neurons, "
                   "removed bins blank", loc="left", fontsize=24)

    for ax in (ax_p, ax_v, ax_h):
        for c in REMOVED_COLOR:
            on = np.where(cause == c)[0]
            for start, stop in contiguous(on):
                ax.axvspan(t[start], t[stop] + BIN_S, color=REMOVED_COLOR[c], lw=0, zorder=0)
    pos_cm = rec["position"][a:b] * CM_PER_DEGREE
    ax_p.plot(t, pos_cm, color=INK, lw=2.5)
    ax_p.axhline(CONTEXT_EXIT_CM, color=INK_SOFT, lw=1.2, ls=(0, (4, 3)))
    ax_p.set_title("position (cm)", loc="left", fontsize=22)
    ax_v.plot(t, rec["velocity"][a:b], color=INK, lw=2)
    ax_v.axhline(1, color=INK_SOFT, lw=1.2, ls=(0, (4, 3)))
    ax_v.set_title("speed (cm/s)", loc="left", fontsize=22)
    ax_v.set_xlabel("Time (s)")
    ax_v.set_xlim(0, t[-1] + BIN_S)
    handles = [Patch(color=REMOVED_COLOR[c], label=CAUSE_LABEL[c]) for c in REMOVED_COLOR]
    fig.align_ylabels(axes)
    fig.legend(handles=handles, loc="outside lower center", ncols=3, fontsize=22)
    save(fig, f"preprocessing_{info.animal}_{TRIALS[0]}-{TRIALS[1]}")


def contiguous(idx: np.ndarray) -> list[tuple[int, int]]:
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    stops = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts, stops, strict=True))


if __name__ == "__main__":
    main()
