"""Pre-processing helpers borrowed from Sebas's ibioml work-in-progress.

The functions in this module are adapted **verbatim** from Sebas's branch
`feat/pre-processing` of `mariburginlab-labPrograms/IBioML` (commit
`402af75`, file `ibioml/preprocess_data.py`).

They live here in `analysis/` rather than being imported from `ibioml`
for two reasons:

1. The thesis pins the `ibioml` submodule to `thesis/audit-baseline` (see
   `specs/decisions.md` D-002) — the audit reproducibility anchor. Moving
   the pin would compromise Phase 1's reproducibility.
2. `feat/pre-processing` is a WIP branch on the lab repo. Until it merges
   to upstream `main`, depending on it directly would tie this thesis to
   a branch that may be rebased / force-pushed / reshaped.

Once Sebas's branch is merged into upstream `main` and the thesis bumps
its submodule pointer accordingly, this module should be deleted and the
imports switched to `from ibioml.preprocess_data import periods_immobility`.
"""

from __future__ import annotations

import numpy as np


def periods_immobility(
    vels_binned,
    thVel,
    binLength,
    thDur,
    trialFinalBin,
    bins_before,
    bins_after,
):
    """Detect bins of prolonged immobility and bins around trial transitions.

    Identifies two classes of bin indices worth excluding from downstream
    preprocessing:

    - **immobility**: bins inside a contiguous stretch where ``velocity < thVel``
      that lasts longer than ``thDur`` seconds. Each stretch is padded by
      ``bins_after`` before its start and ``bins_before`` after its end (the
      asymmetric naming follows the original code; it compensates for
      history-window reach into adjacent bins).
    - **trial_transitions**: bins around each trial boundary, including a
      ~400 ms buffer into the next trial (right after the teleport the
      recording tends to be unstable).

    Parameter order and naming are kept verbatim from the original so the
    two implementations stay swappable when Sebas's branch is merged
    upstream.

    Parameters
    ----------
    vels_binned : array_like
        Velocity per bin. Units of ``thVel`` must match (Sebas's pipeline
        uses cm/s for both).
    thVel : float
        Velocity threshold below which a bin counts as "low velocity".
    binLength : float
        Bin duration in milliseconds.
    thDur : float
        Minimum duration of a low-velocity stretch (in seconds) for it to
        count as immobility.
    trialFinalBin : array_like
        Index of the last bin of each trial. **0-indexed** — the original
        ibioml code loads this as ``np.ravel(...) - 1`` to convert from
        MATLAB. Passing 1-indexed values produces off-by-one errors at
        the trial boundaries.
    bins_before, bins_after : int
        Padding for history-window compatibility.

    Returns
    -------
    immobility : list of int
        Bin indices flagged by the immobility rule.
    trial_transitions : list of int
        Bin indices flagged as trial-transition zones.
    """
    candidates = np.array(np.ravel(vels_binned) < thVel, dtype=int)
    candidates[-1] = 0
    jumps = np.append(candidates[0], np.diff(candidates))
    eventsOn = np.where(jumps == 1)[0]
    eventsOff = np.where(jumps == -1)[0]
    durations = (eventsOff - eventsOn) * binLength / 1000
    included = np.where(durations > thDur)[0]
    immobility = []
    for i in included:
        immobility.extend(
            range(eventsOn[i] - bins_after, eventsOff[i] + bins_before + 1)
        )

    trial_transitions = []
    for trial_idx in range(len(trialFinalBin)):
        start_idx = trialFinalBin[trial_idx] + 1 - bins_before
        if trial_idx == len(trialFinalBin) - 1:
            end_idx = trialFinalBin[-1]
        else:
            end_idx = (
                trialFinalBin[trial_idx]
                + 1
                + int(400 / binLength)
                + bins_after
            )
        trial_transitions.extend(range(start_idx, end_idx + 1))

    return immobility, trial_transitions
