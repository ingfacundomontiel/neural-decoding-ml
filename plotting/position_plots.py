"""
Position plotting functions for neural decoding visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from .utils import (get_default_colors, add_trial_boundaries, 
                   validate_input_data, setup_position_plot_styling)


def plot_position_timeseries(time_bins, position_data, time_bin_size=0.2, 
                           trial_ids=None, labels=None, colors=None, 
                           show_trials=True, title=None, xlabel=None, ylabel=None,
                           figsize=(12, 6), ax=None, **kwargs):
    """
    Plot position data over time, with support for multiple series and trial boundaries.
    
    Parameters
    ----------
    time_bins : array-like
        Array of time bin indices
    position_data : array-like or dict
        Position data. Can be:
        - Single array for one trace
        - Dict {'label': data_array} for multiple traces
    time_bin_size : float, optional
        Size of each time bin in seconds (default: 0.2)
    trial_ids : array-like, optional
        Trial ID for each time bin. If provided and show_trials=True,
        vertical lines will mark trial boundaries
    labels : list, optional
        Custom labels for multiple series. Only used if position_data is not a dict
    colors : list, optional
        Custom colors for multiple series. If None, uses default palette
    show_trials : bool, optional
        Whether to show trial boundaries (default: True)
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label (default: 'Time (s)')
    ylabel : str, optional
        Y-axis label (default: 'Position')
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (12, 6))
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new figure
    **kwargs
        Additional keyword arguments passed to matplotlib plot()
        
    Returns
    -------
    tuple
        (fig, ax) - matplotlib figure and axes objects
        
    Examples
    --------
    # Single trace
    fig, ax = plot_position_timeseries(time_bins, actual_position)
    
    # Multiple model comparison
    position_dict = {'Actual': actual_pos, 'LSTM': lstm_pred, 'MLP': mlp_pred}
    fig, ax = plot_position_timeseries(time_bins, position_dict, trial_ids=trial_ids)
    
    # Custom styling
    fig, ax = plot_position_timeseries(
        time_bins, position_data, 
        colors=['black', 'red'], 
        title='Position Decoding Results'
    )
    """
    
    # Validate input data
    time_bins, position_data, trial_ids = validate_input_data(time_bins, position_data, trial_ids)
    
    # Convert time bins to seconds
    time_seconds = np.asarray(time_bins) * time_bin_size
    
    # Setup figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get colors
    if colors is None:
        colors = get_default_colors()
    
    # Ensure we have enough colors for all series
    n_series = len(position_data)
    if len(colors) < n_series:
        # Extend colors by cycling through the default palette
        base_colors = get_default_colors()
        colors = (colors + base_colors * ((n_series // len(base_colors)) + 1))[:n_series]
    
    # Plot each series
    for i, (label, data) in enumerate(position_data.items()):
        color = colors[i % len(colors)]
        ax.plot(time_seconds, data, label=label, color=color, **kwargs)
    
    # Add trial boundaries if requested
    if show_trials and trial_ids is not None:
        add_trial_boundaries(ax, time_seconds, trial_ids)
    
    # Apply styling
    setup_position_plot_styling(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    
    # Add legend if multiple series
    if len(position_data) > 1:
        ax.legend()
    
    plt.tight_layout()
    return fig, ax


def plot_position_comparison(actual_position, predicted_positions, time_bins, 
                           time_bin_size=0.2, trial_ids=None, model_names=None,
                           show_trials=True, title=None, figsize=(12, 6), **kwargs):
    """
    Convenience function for comparing actual vs predicted position data.
    
    Parameters
    ----------
    actual_position : array-like
        Ground truth position data
    predicted_positions : array-like or dict
        Predicted position data. Can be:
        - Single array for one model
        - Dict {'model_name': predictions} for multiple models
        - List of arrays for multiple models (requires model_names)
    time_bins : array-like
        Array of time bin indices
    time_bin_size : float, optional
        Size of each time bin in seconds (default: 0.2)
    trial_ids : array-like, optional
        Trial ID for each time bin
    model_names : list, optional
        Names for models if predicted_positions is a list
    show_trials : bool, optional
        Whether to show trial boundaries (default: True)
    title : str, optional
        Plot title (default: 'Position: Actual vs Predicted')
    figsize : tuple, optional
        Figure size (default: (12, 6))
    **kwargs
        Additional arguments passed to plot_position_timeseries()
        
    Returns
    -------
    tuple
        (fig, ax) - matplotlib figure and axes objects
    """
    
    # Prepare position data dictionary
    position_dict = {'Actual': actual_position}
    
    # Handle different formats of predicted_positions
    if isinstance(predicted_positions, dict):
        position_dict.update(predicted_positions)
    elif isinstance(predicted_positions, (list, tuple)):
        if model_names is None:
            model_names = [f'Model_{i+1}' for i in range(len(predicted_positions))]
        for name, pred in zip(model_names, predicted_positions):
            position_dict[name] = pred
    else:
        # Single array
        model_name = model_names[0] if model_names else 'Predicted'
        position_dict[model_name] = predicted_positions
    
    # Set default title
    if title is None:
        title = 'Position: Actual vs Predicted'
    
    # Use main plotting function
    return plot_position_timeseries(
        time_bins, position_dict, time_bin_size=time_bin_size,
        trial_ids=trial_ids, show_trials=show_trials, 
        title=title, figsize=figsize, **kwargs
    ) 