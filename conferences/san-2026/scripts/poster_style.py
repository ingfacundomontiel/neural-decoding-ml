"""Shared style for the SAN 2026 poster figures (plan.md → "Style guide").

Every figure script does ``from poster_style import *`` (or imports names), calls
``setup()`` once, builds its figure with ``figsize=cm(w, h)`` — the physical size
of its slot on the poster — and saves it with ``save(fig, name)``. Sizes in the
mplstyle are printed points, so nothing may be rescaled after export.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
POSTER = HERE.parent
REPO = POSTER.parent.parent
FIG_DIR = POSTER / "figures"
sys.path.insert(0, str(REPO))

# --- palette (semantic: one meaning per colour) ------------------------------
INK = "#1E1E1E"
INK_SOFT = "#5A5A5A"
REGION_COLOR = {"Pir": "#0B63A5", "CA3": "#8E3B8F", "DG": "#C0581B"}
# Naive = same hue, lighter; used for fills/lines of naive recordings.
REGION_LIGHT = {"Pir": "#7FB1DA", "CA3": "#C99BCB", "DG": "#EBAA83"}
REGION_NAME = {"Pir": "Piriform cortex", "CA3": "CA3", "DG": "DG"}
CTX_REWARDED = "#5FA35A"   # the VR's green wall
CTX_UNREWARDED = "#9A9A9A"
CONTEXT_ZONE = "#FFF3C4"
CONTEXT_ZONE_EDGE = "#E3C75A"
# Removed-bin causes in the preprocessing figure (neutral/warm, not region hues).
REMOVED_COLOR = {"transition": "#D9D9D9", "immobility": "#F2C6C2", "tail": "#C9DDEA"}

# --- corridor landmarks (analysis/loaders.py, D-017) --------------------------
CONTEXT_ENTRY_CM = 44.7
CONTEXT_EXIT_CM = 62.9

# --- datasets -----------------------------------------------------------------
REGION_ORDER = ("Pir", "CA3", "DG")
COHORT_ORDER = ("naive", "expert")   # before → after learning


def cm(w: float, h: float) -> tuple[float, float]:
    """Figure size in inches from centimetres."""
    return (w / 2.54, h / 2.54)


def setup() -> None:
    """Register the bundled IBM Plex Sans files and load the poster style."""
    for ttf in sorted((POSTER / "assets" / "fonts").glob("*.ttf")):
        font_manager.fontManager.addfont(str(ttf))
    plt.style.use(HERE / "poster.mplstyle")


def save(fig: plt.Figure, name: str) -> Path:
    """Save *fig* as SVG (poster) + PNG preview under figures/."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.svg"
    fig.savefig(out)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=60)
    plt.close(fig)
    return out


def shade_context(ax, orientation: str = "x", **kw) -> None:
    """Pale band over the odour-context zone on a position axis."""
    span = ax.axvspan if orientation == "x" else ax.axhspan
    span(CONTEXT_ENTRY_CM, CONTEXT_EXIT_CM, color=CONTEXT_ZONE, zorder=0, lw=0, **kw)


def dataset_sort_key(region: str, cohort: str, animal: str) -> tuple:
    return (REGION_ORDER.index(region), COHORT_ORDER.index(cohort), animal)
