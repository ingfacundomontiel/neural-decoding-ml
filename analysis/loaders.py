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
a wheel) point to degrees. Confirmed correct by the director, 2026-07-14.
"""

from __future__ import annotations

from pathlib import Path

from scipy.io import loadmat

# Wheel-encoder degrees → corridor centimetres. See module docstring.
CM_PER_DEGREE = 0.1222

# Virtual-corridor geometry, in cm. Context entry/exit confirmed with the
# director (2026-08-26; supersedes the 42 / 59 cm working estimates of
# 2026-06-12). The reward zone starts where the context ends. The natural
# end of the corridor is not documented by the lab; until it is, the context
# exit is the last known landmark and everything beyond it is treated as
# "post-context" — reward consumption, licking drift, or the animal running
# past the end. See docs/trial-structure.md and exploration/01-trial-structure.py.
CONTEXT_START_CM = 44.7   # animal enters the rewarded/non-rewarded context
CONTEXT_END_CM = 62.9     # exits the context; reward zone begins here
REWARD_START_CM = CONTEXT_END_CM
CORRIDOR_END_CM = CONTEXT_END_CM  # working assumption — see note above

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
