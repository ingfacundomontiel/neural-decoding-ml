"""
Trial Position Plotter Script

This script allows you to select specific trials from your preprocessed neural decoding data 
and visualize the mouse position over time.

Features:
- Load preprocessed data
- Select specific trials by ID
- Plot position with trial boundaries
- Customizable trial selection options
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_position_timeseries


def load_preprocessed_data():
    """
    Load preprocessed neural decoding data.
    
    Returns
    -------
    tuple
        (X, y, trial_ids, position, unique_trials)
    """
    print("Loading preprocessed data...")
    with open('processed-datasets/L5_bins200ms_withCtxt_preprocessed.pickle', 'rb') as f:
        X, y, trial_ids = pickle.load(f)

    # Extract position data (first column if multi-dimensional)
    position = y[:, 0] if y.ndim > 1 else y
    unique_trials = np.unique(trial_ids)

    print("Data loaded successfully!")
    print(f"- Neural data shape: {X.shape}")
    print(f"- Position data shape: {position.shape}")
    print(f"- Trial IDs shape: {trial_ids.shape}")
    print(f"- Number of unique trials: {len(unique_trials)}")
    print(f"- Trial ID range: {np.min(trial_ids)} to {np.max(trial_ids)}")
    print(f"- Total time: {len(y) * 0.2:.1f} seconds")
    print(f"\nAvailable trial IDs: {unique_trials}")

    return X, y, trial_ids, position, unique_trials


def show_trial_statistics(trial_ids, unique_trials):
    """
    Display statistics about trial lengths.
    
    Parameters
    ----------
    trial_ids : array
        Trial ID for each time bin
    unique_trials : array
        Array of unique trial IDs
    """
    trial_lengths = []
    for trial_id in unique_trials:
        trial_mask = trial_ids == trial_id
        trial_length = np.sum(trial_mask)
        trial_lengths.append(trial_length)

    trial_lengths = np.array(trial_lengths)
    print(f"\nTrial length statistics:")
    print(f"- Mean trial length: {np.mean(trial_lengths):.1f} time bins ({np.mean(trial_lengths) * 0.2:.1f} seconds)")
    print(f"- Min trial length: {np.min(trial_lengths)} time bins ({np.min(trial_lengths) * 0.2:.1f} seconds)")
    print(f"- Max trial length: {np.max(trial_lengths)} time bins ({np.max(trial_lengths) * 0.2:.1f} seconds)")


def select_and_validate_trials(selected_trials, unique_trials, trial_ids, position):
    """
    Validate selected trials and extract corresponding data.
    
    Parameters
    ----------
    selected_trials : list
        List of trial IDs to plot
    unique_trials : array
        Array of all available trial IDs
    trial_ids : array
        Trial ID for each time bin
    position : array
        Position data
        
    Returns
    -------
    tuple
        (selected_time_bins, selected_position, selected_trial_ids, valid_trials)
    """
    print(f"Selected trials: {selected_trials}")
    print(f"Number of trials to plot: {len(selected_trials)}")

    # Validate selected trials
    valid_trials = []
    for trial in selected_trials:
        if trial in unique_trials:
            valid_trials.append(trial)
        else:
            print(f"Warning: Trial {trial} not found in data")

    selected_trials = valid_trials
    print(f"Valid trials to plot: {selected_trials}")

    if len(selected_trials) == 0:
        print("No valid trials selected!")
        return None, None, None, []

    # Get data for selected trials
    selected_mask = np.isin(trial_ids, selected_trials)
    selected_time_bins = np.arange(len(trial_ids))[selected_mask]
    selected_position = position[selected_mask]
    selected_trial_ids = trial_ids[selected_mask]

    print(f"\nData for selected trials:")
    print(f"- Number of time bins: {len(selected_time_bins)}")
    print(f"- Duration: {len(selected_time_bins) * 0.2:.1f} seconds")
    print(f"- Position range: {np.min(selected_position):.1f} to {np.max(selected_position):.1f}")

    # Show trial boundaries
    for trial in selected_trials:
        trial_mask = selected_trial_ids == trial
        trial_bins = np.sum(trial_mask)
        trial_duration = trial_bins * 0.2
        print(f"- Trial {trial}: {trial_bins} bins ({trial_duration:.1f}s)")

    return selected_time_bins, selected_position, selected_trial_ids, selected_trials


def plot_selected_trials(selected_time_bins, selected_position, selected_trial_ids, selected_trials, 
                        figsize=(15, 6), save_plot=False, filename=None):
    """
    Plot the position data for selected trials.
    
    Parameters
    ----------
    selected_time_bins : array
        Time bin indices for selected trials
    selected_position : array
        Position data for selected trials
    selected_trial_ids : array
        Trial IDs for selected trials
    selected_trials : list
        List of trial IDs being plotted
    figsize : tuple, optional
        Figure size (default: (15, 6))
    save_plot : bool, optional
        Whether to save the plot (default: False)
    filename : str, optional
        Filename for saving the plot
    
    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axes objects
    """
    if len(selected_trials) == 0:
        print("No valid trials to plot.")
        return None, None

    print(f"Plotting position data for trials: {selected_trials}")
    
    # Create the plot
    fig, ax = plot_position_timeseries(
        time_bins=selected_time_bins,
        position_data=selected_position,
        trial_ids=selected_trial_ids,
        time_bin_size=0.2,
        title=f"Mouse Position - Trials {selected_trials}",
        show_trials=True,
        figsize=figsize
    )
    
    # Add some additional formatting
    ax.set_ylabel("Position")
    ax.set_xlabel("Time (seconds)")
    
    # Add trial labels at the top
    for trial in selected_trials:
        trial_mask = selected_trial_ids == trial
        trial_indices = np.where(trial_mask)[0]
        if len(trial_indices) > 0:
            # Get the middle time point of the trial
            middle_idx = trial_indices[len(trial_indices) // 2]
            middle_time = selected_time_bins[middle_idx] * 0.2
            
            # Add trial label
            ax.text(middle_time, ax.get_ylim()[1] * 0.95, f'Trial {trial}', 
                   ha='center', va='top', fontsize=10, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        if filename is None:
            filename = f"trial_position_plot_{'-'.join(map(str, selected_trials))}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
    
    plt.show()
    
    # Print some statistics
    print(f"\nPlot Statistics:")
    print(f"- Total duration plotted: {len(selected_time_bins) * 0.2:.1f} seconds")
    print(f"- Position range: {np.min(selected_position):.1f} to {np.max(selected_position):.1f}")
    print(f"- Number of trials: {len(selected_trials)}")
    
    return fig, ax


def main():
    """
    Main function to run the trial position plotter.
    """
    print("=" * 60)
    print("TRIAL POSITION PLOTTER")
    print("=" * 60)
    
    # Load data
    X, y, trial_ids, position, unique_trials = load_preprocessed_data()
    show_trial_statistics(trial_ids, unique_trials)
    
    print("\n" + "=" * 60)
    print("TRIAL SELECTION")
    print("=" * 60)
    
    # ========================================
    # TRIAL SELECTION - MODIFY THIS SECTION
    # ========================================
    
    # Option 1: Select specific trial IDs
    selected_trials = [34, 35, 36]  # Example: first 3 trials
    
    # Option 2: Select a range of trials
    # selected_trials = list(range(34, 40))  # Trials 34 to 39
    
    # Option 3: Select random trials
    # np.random.seed(42)
    # selected_trials = np.random.choice(unique_trials, size=5, replace=False).tolist()
    
    # Option 4: Select trials by index (first N trials)
    # selected_trials = unique_trials[:5].tolist()  # First 5 trials
    
    # Option 5: Select all trials (warning: may be slow for large datasets)
    # selected_trials = unique_trials.tolist()
    
    # ========================================
    
    # Validate and extract data for selected trials
    selected_time_bins, selected_position, selected_trial_ids, valid_trials = select_and_validate_trials(
        selected_trials, unique_trials, trial_ids, position
    )
    
    print("\n" + "=" * 60)
    print("PLOTTING")
    print("=" * 60)
    
    # Plot the selected trials
    fig, ax = plot_selected_trials(
        selected_time_bins, selected_position, selected_trial_ids, valid_trials,
        figsize=(15, 6),
        save_plot=False,  # Set to True to save the plot
        filename=None     # Custom filename for saving
    )
    
    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)


def plot_custom_trials(trial_list, figsize=(15, 6), save_plot=False, filename=None):
    """
    Convenience function to quickly plot specific trials.
    
    Parameters
    ----------
    trial_list : list
        List of trial IDs to plot
    figsize : tuple, optional
        Figure size (default: (15, 6))
    save_plot : bool, optional
        Whether to save the plot (default: False)
    filename : str, optional
        Filename for saving the plot
        
    Examples
    --------
    # Plot specific trials
    plot_custom_trials([34, 35, 36])
    
    # Plot and save
    plot_custom_trials([50, 51, 52], save_plot=True, filename="trials_50-52.png")
    """
    # Load data
    X, y, trial_ids, position, unique_trials = load_preprocessed_data()
    
    # Select and validate trials
    selected_time_bins, selected_position, selected_trial_ids, valid_trials = select_and_validate_trials(
        trial_list, unique_trials, trial_ids, position
    )
    
    # Plot
    return plot_selected_trials(
        selected_time_bins, selected_position, selected_trial_ids, valid_trials,
        figsize=figsize, save_plot=save_plot, filename=filename
    )


if __name__ == "__main__":
    main()


# ===== USAGE EXAMPLES =====
"""
# Example usage in other scripts:

from trial_position_plotter import plot_custom_trials, load_preprocessed_data

# Quick plot of specific trials
plot_custom_trials([34, 35, 36])

# Load data and work with it directly
X, y, trial_ids, position, unique_trials = load_preprocessed_data()

# Useful trial selection examples:
selected_trials = [34, 35, 36]  # Specific trials
selected_trials = list(range(50, 60))  # Trials 50-59
selected_trials = unique_trials[::10].tolist()  # Every 10th trial
selected_trials = unique_trials[:5].tolist()    # First 5 trials
selected_trials = unique_trials[-5:].tolist()   # Last 5 trials

# Random selection
np.random.seed(42)
selected_trials = np.random.choice(unique_trials, size=8, replace=False).tolist()
""" 