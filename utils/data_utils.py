import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union


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