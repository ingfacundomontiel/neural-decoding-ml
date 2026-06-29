"""Loaders for the lab's `.mat` recordings.

The raw `.mat` files under `data/raw/` follow the structure documented in
`data/raw/README.md` (File format section) and `specs/research.md` (Common
data model).

Position-to-cm conversion
-------------------------
The `position` variable in every `.mat` is in **wheel-encoder degrees**.
The conversion factor to corridor centimetres was copied from
`marinburginlab-labPrograms/.../san/all-datasets-plot-predictions.py`
line 222 (`cm_per_degree = 0.1222`). The comment in that file mentions
"mm → cm" but the variable name and the upstream rig (rotary encoder on
a wheel) point to degrees. Confirm with the director if this matters.
"""

from __future__ import annotations

from pathlib import Path

from scipy.io import loadmat

# Wheel-encoder degrees → corridor centimetres. See module docstring.
CM_PER_DEGREE = 0.1222

# Virtual-corridor geometry, in cm, as reported by Facundo's collaborators
# (verified against the lab's experimental setup, 2026-06-12).
# Anything beyond CORRIDOR_END_CM is "drift" — either the rig extending the
# corridor while the animal licks, or the animal continuing to run past the
# expected end in unrewarded trials. See exploration/01-trial-structure.py.
CONTEXT_START_CM = 42.0   # animal enters the rewarded/non-rewarded context
CONTEXT_END_CM = 59.0     # exits the context
REWARD_END_CM = 60.0      # end of the reward zone (1-2 cm wide, just after context)
CORRIDOR_END_CM = 60.0    # the "natural" end of the virtual corridor; position > this is drift

# Variables we expect in every well-formed recording `.mat`.
EXPECTED_VARS = (
    "neuronActivity",
    "position",
    "rewCtxt",
    "rewOdor",
    "goResponse",
    "dPrime",
    "criterion",
    "trialDurationInBins",
    "trialFinalBin",
)


def load_mat(path: str | Path) -> dict:
    """Load a recording `.mat` and return a dict of squeezed numpy arrays.

    Drops MATLAB metadata keys (those starting with `__`). Does not enforce
    variable presence — use ``check_expected(d)`` for that.
    """
    mat = loadmat(str(path), squeeze_me=True)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def check_expected(d: dict) -> dict:
    """Report which EXPECTED_VARS are present, missing, or extras."""
    present = [k for k in EXPECTED_VARS if k in d]
    missing = [k for k in EXPECTED_VARS if k not in d]
    extras = [k for k in d.keys() if k not in EXPECTED_VARS]
    return {"present": present, "missing": missing, "extras": extras}


def position_in_cm(position_deg):
    """Convert position from wheel-encoder degrees to corridor centimetres."""
    return position_deg * CM_PER_DEGREE
