import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union
import warnings


class NeuralDataset(Dataset):
    """
    Custom dataset for neural decoding data that can handle both 
    flattened (MLP) and sequential (RNN/LSTM) input formats.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, flatten_temporal: bool = False):
        """
        Args:
            X: Neural data with temporal windows [samples, time_bins, features]
            y: Target positions [samples]
            flatten_temporal: If True, flatten temporal dimension for MLP input
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.flatten_temporal = flatten_temporal
        
        if flatten_temporal:
            # Flatten temporal dimension: [samples, time_bins * features]
            self.X = self.X.view(self.X.shape[0], -1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def get_input_size(self) -> int:
        """Return the input size for model configuration"""
        if self.flatten_temporal:
            return self.X.shape[1]  # Flattened features
        else:
            return self.X.shape[2]  # Features per time step
    
    def get_sequence_length(self) -> int:
        """Return sequence length (only meaningful for sequential models)"""
        if self.flatten_temporal:
            return None
        else:
            return self.X.shape[1]  # Time bins


def analyze_trial_characteristics(trial_ids: np.ndarray, y: np.ndarray) -> dict:
    """
    Analyze characteristics of trials in the dataset
    
    Args:
        trial_ids: Trial identifiers for each sample
        y: Target values (positions)
    
    Returns:
        dict: Comprehensive trial analysis
    """
    unique_trials = np.unique(trial_ids)
    trial_info = []
    
    for trial_id in unique_trials:
        trial_mask = trial_ids == trial_id
        trial_length = np.sum(trial_mask)
        trial_positions = y[trial_mask].flatten()
        total_movement = np.sum(np.abs(np.diff(trial_positions))) if len(trial_positions) > 1 else 0
        
        trial_info.append({
            'id': trial_id,
            'length': trial_length,
            'duration_sec': trial_length * 0.2,
            'start_pos': trial_positions[0],
            'end_pos': trial_positions[-1],
            'total_movement': total_movement,
            'pos_range': np.max(trial_positions) - np.min(trial_positions)
        })
    
    # Calculate summary statistics
    lengths = [t['length'] for t in trial_info]
    durations = [t['duration_sec'] for t in trial_info]
    movements = [t['total_movement'] for t in trial_info]
    
    return {
        'num_trials': len(unique_trials),
        'trial_info': trial_info,
        'lengths': lengths,
        'durations': durations,
        'movements': movements,
        'total_samples': len(trial_ids),
        'length_stats': {
            'min': np.min(lengths),
            'max': np.max(lengths),
            'mean': np.mean(lengths),
            'std': np.std(lengths)
        },
        'duration_stats': {
            'min': np.min(durations),
            'max': np.max(durations),
            'mean': np.mean(durations),
            'std': np.std(durations)
        }
    }


def create_trial_based_data_loaders(X: np.ndarray, y: np.ndarray, trial_ids: np.ndarray,
                                  model_type: str,
                                  batch_size: int = 64,
                                  validation_split: float = 0.2,
                                  test_split: float = 0.1,
                                  shuffle: bool = True,
                                  random_seed: int = 42,
                                  stratify_by_duration: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train, validation, and test data loaders based on TRIAL splits (not individual time bins)
    
    Args:
        X: Neural data [samples, time_bins, features]
        y: Target positions [samples]
        trial_ids: Trial identifiers for each sample [samples]
        model_type: 'MLP', 'RNN', or 'LSTM'
        batch_size: Batch size for data loaders
        validation_split: Fraction of trials for validation
        test_split: Fraction of trials for testing
        shuffle: Whether to shuffle the data within loaders
        random_seed: Random seed for reproducibility
        stratify_by_duration: Whether to stratify splits by trial duration
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, split_info)
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Analyze trial characteristics
    trial_analysis = analyze_trial_characteristics(trial_ids, y)
    
    print(f"📊 TRIAL-BASED SPLITTING ANALYSIS")
    print(f"Total trials: {trial_analysis['num_trials']}")
    print(f"Total samples: {trial_analysis['total_samples']}")
    print(f"Trial duration: {trial_analysis['duration_stats']['mean']:.1f} ± {trial_analysis['duration_stats']['std']:.1f} seconds")
    print(f"Trial length: {trial_analysis['length_stats']['mean']:.1f} ± {trial_analysis['length_stats']['std']:.1f} bins")
    
    # Get unique trials and their characteristics
    unique_trials = np.unique(trial_ids)
    trial_info = trial_analysis['trial_info']
    
    # Split trials (not individual samples)
    if stratify_by_duration:
        # Sort trials by duration for stratified splitting
        trial_info_sorted = sorted(trial_info, key=lambda x: x['duration_sec'])
        trials_sorted = [t['id'] for t in trial_info_sorted]
    else:
        trials_sorted = unique_trials.copy()
        np.random.shuffle(trials_sorted)
    
    # Calculate split sizes for trials
    n_trials = len(trials_sorted)
    n_test_trials = max(1, int(n_trials * test_split))
    n_val_trials = max(1, int(n_trials * validation_split))
    n_train_trials = n_trials - n_test_trials - n_val_trials
    
    if stratify_by_duration:
        # Distribute trials across splits to balance duration
        test_trials = trials_sorted[::n_trials//n_test_trials][:n_test_trials]
        remaining_trials = [t for t in trials_sorted if t not in test_trials]
        
        val_trials = remaining_trials[::len(remaining_trials)//n_val_trials][:n_val_trials]
        train_trials = [t for t in remaining_trials if t not in val_trials]
    else:
        # Simple sequential split
        test_trials = trials_sorted[:n_test_trials]
        val_trials = trials_sorted[n_test_trials:n_test_trials + n_val_trials]
        train_trials = trials_sorted[n_test_trials + n_val_trials:]
    
    # Collect sample indices for each split
    train_indices = np.where(np.isin(trial_ids, train_trials))[0]
    val_indices = np.where(np.isin(trial_ids, val_trials))[0]
    test_indices = np.where(np.isin(trial_ids, test_trials))[0]
    
    # Report split statistics
    split_info = {
        'train_trials': len(train_trials),
        'val_trials': len(val_trials),
        'test_trials': len(test_trials),
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'test_samples': len(test_indices),
        'train_trial_ids': train_trials,
        'val_trial_ids': val_trials,
        'test_trial_ids': test_trials
    }
    
    print(f"\n📋 TRIAL SPLIT RESULTS:")
    print(f"Train: {len(train_trials)} trials ({len(train_indices)} samples, {len(train_indices)/len(trial_ids)*100:.1f}%)")
    print(f"Val:   {len(val_trials)} trials ({len(val_indices)} samples, {len(val_indices)/len(trial_ids)*100:.1f}%)")
    print(f"Test:  {len(test_trials)} trials ({len(test_indices)} samples, {len(test_indices)/len(trial_ids)*100:.1f}%)")
    
    # Calculate duration balance
    train_duration = sum(t['duration_sec'] for t in trial_info if t['id'] in train_trials)
    val_duration = sum(t['duration_sec'] for t in trial_info if t['id'] in val_trials)
    test_duration = sum(t['duration_sec'] for t in trial_info if t['id'] in test_trials)
    total_duration = train_duration + val_duration + test_duration
    
    print(f"\n⏱️ DURATION BALANCE:")
    print(f"Train: {train_duration:.1f}s ({train_duration/total_duration*100:.1f}%)")
    print(f"Val:   {val_duration:.1f}s ({val_duration/total_duration*100:.1f}%)")
    print(f"Test:  {test_duration:.1f}s ({test_duration/total_duration*100:.1f}%)")
    
    # Determine if we need to flatten temporal dimension
    flatten_temporal = model_type.upper() == 'MLP'
    
    # Create datasets for each split
    train_dataset = NeuralDataset(X[train_indices], y[train_indices], flatten_temporal=flatten_temporal)
    val_dataset = NeuralDataset(X[val_indices], y[val_indices], flatten_temporal=flatten_temporal)
    test_dataset = NeuralDataset(X[test_indices], y[test_indices], flatten_temporal=flatten_temporal)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, split_info


def create_data_loaders(X: np.ndarray, y: np.ndarray, 
                       model_type: str,
                       batch_size: int = 64,
                       validation_split: float = 0.2,
                       test_split: float = 0.1,
                       shuffle: bool = True,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders for different model types
    
    Args:
        X: Neural data [samples, time_bins, features]
        y: Target positions [samples]
        model_type: 'MLP', 'RNN', or 'LSTM'
        batch_size: Batch size for data loaders
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        shuffle: Whether to shuffle the data
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Determine if we need to flatten temporal dimension
    flatten_temporal = model_type.upper() == 'MLP'
    
    # Create dataset
    dataset = NeuralDataset(X, y, flatten_temporal=flatten_temporal)
    
    # Calculate split sizes
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size - test_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def get_data_info(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Get comprehensive information about the dataset
    
    Args:
        X: Neural data [samples, time_bins, features] 
        y: Target positions [samples]
    
    Returns:
        dict: Dataset information
    """
    return {
        'num_samples': X.shape[0],
        'num_time_bins': X.shape[1],
        'num_features': X.shape[2],
        'flattened_input_size': X.shape[1] * X.shape[2],
        'position_range': (float(np.min(y)), float(np.max(y))),
        'position_mean': float(np.mean(y)),
        'position_std': float(np.std(y)),
        'X_shape': X.shape,
        'y_shape': y.shape
    }


def prepare_data_for_model(X: np.ndarray, y: np.ndarray, model_type: str) -> Tuple[int, int]:
    """
    Analyze data and return the appropriate input size and sequence length for model configuration
    
    Args:
        X: Neural data [samples, time_bins, features]
        y: Target positions [samples] 
        model_type: 'MLP', 'RNN', or 'LSTM'
    
    Returns:
        Tuple of (input_size, sequence_length)
    """
    if model_type.upper() == 'MLP':
        # For MLP: input_size is flattened features, sequence_length is not used
        input_size = X.shape[1] * X.shape[2]  # time_bins * features
        sequence_length = None
    else:
        # For RNN/LSTM: input_size is features per time step
        input_size = X.shape[2]  # features
        sequence_length = X.shape[1]  # time_bins
    
    return input_size, sequence_length 