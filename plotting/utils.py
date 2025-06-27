"""
Utility functions for plotting neural decoding data.
"""

import numpy as np
import matplotlib.pyplot as plt


def get_default_colors():
    """
    Get the default color palette for plotting multiple series.
    
    Returns
    -------
    list
        List of color strings in order: black, blue, red
    """
    return ['black', 'blue', 'red']


def add_trial_boundaries(ax, time_data, trial_ids, alpha=0.3, color='gray', linestyle='--'):
    """
    Add vertical lines at trial boundaries to a matplotlib axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add trial boundaries to
    time_data : array-like
        Time values (in seconds) corresponding to each data point
    trial_ids : array-like
        Trial ID for each time point
    alpha : float, optional
        Transparency of the boundary lines (default: 0.3)
    color : str, optional
        Color of the boundary lines (default: 'gray')
    linestyle : str, optional
        Style of the boundary lines (default: '--')
    """
    if trial_ids is None or len(trial_ids) == 0:
        return
    
    # Find indices where trial changes
    trial_changes = np.where(np.diff(trial_ids) != 0)[0]
    
    # Add vertical lines at trial boundaries
    for boundary_idx in trial_changes:
        # boundary_idx is the last index of the previous trial
        # so boundary_idx + 1 is the start of the next trial
        if boundary_idx + 1 < len(time_data):
            boundary_time = time_data[boundary_idx + 1]
            ax.axvline(x=boundary_time, color=color, linestyle=linestyle, 
                      alpha=alpha, linewidth=1)


def validate_input_data(time_bins, position_data, trial_ids=None):
    """
    Validate input data for position plotting functions.
    
    Parameters
    ----------
    time_bins : array-like
        Time bin indices
    position_data : array-like or dict
        Position data (single array or dict of arrays)
    trial_ids : array-like, optional
        Trial IDs corresponding to each time bin
        
    Returns
    -------
    tuple
        (validated_time_bins, validated_position_data, validated_trial_ids)
        
    Raises
    ------
    ValueError
        If input data dimensions don't match or are invalid
    """
    time_bins = np.asarray(time_bins)
    
    # Handle position_data - convert to dict format for consistency
    if isinstance(position_data, dict):
        validated_position_data = position_data
        # Check that all arrays have same length as time_bins
        for label, data in position_data.items():
            data = np.asarray(data)
            if len(data) != len(time_bins):
                raise ValueError(f"Length mismatch: time_bins ({len(time_bins)}) vs {label} data ({len(data)})")
    else:
        # Single array - convert to dict
        position_data = np.asarray(position_data)
        if len(position_data) != len(time_bins):
            raise ValueError(f"Length mismatch: time_bins ({len(time_bins)}) vs position_data ({len(position_data)})")
        validated_position_data = {'Position': position_data}
    
    # Validate trial_ids if provided
    if trial_ids is not None:
        trial_ids = np.asarray(trial_ids)
        if len(trial_ids) != len(time_bins):
            raise ValueError(f"Length mismatch: time_bins ({len(time_bins)}) vs trial_ids ({len(trial_ids)})")
    
    return time_bins, validated_position_data, trial_ids


def setup_position_plot_styling(ax, title=None, xlabel=None, ylabel=None):
    """
    Apply standard styling to position plots.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to style
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label (default: 'Time (s)')
    ylabel : str, optional
        Y-axis label (default: 'Position')
    """
    if xlabel is None:
        xlabel = 'Time (s)'
    if ylabel is None:
        ylabel = 'Position'
        
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title is not None:
        ax.set_title(title)
    
    ax.grid(True, alpha=0.3, axis='y')  # Only horizontal grid lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False) 