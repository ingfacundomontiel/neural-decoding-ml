"""
Plotting module for neural decoding visualizations.

This module provides functions for visualizing neural decoding data,
including position timeseries, model comparisons, and trial-based analysis.
"""

from .position_plots import plot_position_timeseries, plot_position_comparison
from .utils import get_default_colors, add_trial_boundaries

__all__ = [
    'plot_position_timeseries',
    'plot_position_comparison',
    'get_default_colors', 
    'add_trial_boundaries'
] 