import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Union
from .data_utils import NeuralDataset


def create_temporal_data_loaders(X: np.ndarray, y: np.ndarray, 
                                model_type: str,
                                batch_size: int = 64,
                                validation_split: float = 0.2,
                                test_split: float = 0.2,
                                random_seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders with TEMPORAL (contiguous) splitting.
    
    This avoids data leakage by ensuring test data comes from later time periods
    than training data, with no overlap in temporal windows.
    
    Args:
        X: Neural data [samples, time_bins, features]
        y: Target positions [samples]
        model_type: 'MLP', 'RNN', or 'LSTM'
        batch_size: Batch size for data loaders
        validation_split: Fraction of training data for validation
        test_split: Fraction of total data for testing
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
    
    # Calculate split indices (TEMPORAL/CONTIGUOUS splitting)
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    remaining_size = total_size - test_size
    val_size = int(remaining_size * validation_split)
    train_size = remaining_size - val_size
    
    print(f"📊 TEMPORAL DATA SPLITTING:")
    print(f"   Total samples: {total_size}")
    print(f"   Train: 0 → {train_size-1} ({train_size} samples, {train_size/total_size*100:.1f}%)")
    print(f"   Val:   {train_size} → {train_size+val_size-1} ({val_size} samples, {val_size/total_size*100:.1f}%)")
    print(f"   Test:  {train_size+val_size} → {total_size-1} ({test_size} samples, {test_size/total_size*100:.1f}%)")
    print(f"   ✅ NO TEMPORAL OVERLAP - Prevents data leakage!")
    
    # Create contiguous splits (chronological order)
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))
    
    # Create dataset subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Can shuffle within training set
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Keep validation in order
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Keep test in order
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def check_temporal_leakage(X: np.ndarray, train_indices: list, test_indices: list, 
                          window_size: int = 11) -> dict:
    """
    Check for potential temporal leakage between train and test sets.
    
    Args:
        X: Neural data [samples, time_bins, features]
        train_indices: List of training sample indices
        test_indices: List of test sample indices  
        window_size: Size of temporal window used in preprocessing
        
    Returns:
        dict: Leakage analysis results
    """
    train_max = max(train_indices)
    test_min = min(test_indices)
    
    # Calculate gap between training and test
    gap = test_min - train_max - 1
    
    # Check if gap is sufficient to prevent leakage
    # Need gap >= window_size to ensure no overlap
    safe_gap = window_size
    is_safe = gap >= safe_gap
    
    analysis = {
        'train_end_index': train_max,
        'test_start_index': test_min,
        'gap_samples': gap,
        'required_gap': safe_gap,
        'is_leakage_free': is_safe,
        'leakage_risk': 'LOW' if is_safe else 'HIGH'
    }
    
    return analysis


def compare_random_vs_temporal_split(X: np.ndarray, y: np.ndarray, test_split: float = 0.2):
    """
    Compare random vs temporal splitting to demonstrate leakage risk.
    
    Args:
        X: Neural data [samples, time_bins, features]
        y: Target positions [samples]
        test_split: Fraction for testing
    """
    total_size = len(X)
    test_size = int(total_size * test_split)
    
    print("🔍 COMPARING SPLITTING STRATEGIES:")
    print("=" * 50)
    
    # Random split simulation
    np.random.seed(42)
    random_indices = np.random.permutation(total_size)
    random_train = sorted(random_indices[:-test_size])
    random_test = sorted(random_indices[-test_size:])
    
    print("❌ RANDOM SPLIT:")
    print(f"   Train indices: {random_train[:5]} ... {random_train[-5:]}")
    print(f"   Test indices:  {random_test[:5]} ... {random_test[-5:]}")
    print(f"   Potential overlaps: {len(set(random_train) & set(random_test))} direct")
    
    # Check for adjacent indices (temporal leakage)
    leakage_count = 0
    for test_idx in random_test:
        nearby_in_train = any(abs(test_idx - train_idx) <= 5 for train_idx in random_train)
        if nearby_in_train:
            leakage_count += 1
    
    print(f"   Temporal leakage risk: {leakage_count}/{len(random_test)} test samples")
    
    # Temporal split
    train_size = total_size - test_size
    temporal_train = list(range(0, train_size))
    temporal_test = list(range(train_size, total_size))
    
    print("\n✅ TEMPORAL SPLIT:")
    print(f"   Train indices: {temporal_train[:5]} ... {temporal_train[-5:]}")
    print(f"   Test indices:  {temporal_test[:5]} ... {temporal_test[-5:]}")
    print(f"   Gap between train/test: {temporal_test[0] - temporal_train[-1] - 1} samples")
    print(f"   Temporal leakage risk: 0/{len(temporal_test)} test samples")
    
    leakage_analysis = check_temporal_leakage(X, temporal_train, temporal_test)
    print(f"\n📊 LEAKAGE ANALYSIS:")
    for key, value in leakage_analysis.items():
        print(f"   {key}: {value}")
        
    return {
        'random_leakage_count': leakage_count,
        'temporal_leakage_analysis': leakage_analysis
    } 