#!/usr/bin/env python3
"""
Run corrected baseline experiment with proper temporal data splitting.

This version fixes the critical data leakage issue by using contiguous
temporal splits instead of random splits.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from training.baseline_trainer_temporal import BaselineTrainerTemporal


def main():
    """Run corrected baseline experiment with temporal splitting"""
    
    print("🚨 CORRECTED BASELINE EXPERIMENT (TEMPORAL SPLITTING)")
    print("=" * 60)
    print("This experiment fixes the data leakage issue in the original baseline.")
    print("Results should show more realistic (lower) performance for LSTM models.")
    print()
    
    # Initialize trainer with temporal splitting
    trainer = BaselineTrainerTemporal(
        data_path_flat='processed-datasets/L5_bins200ms_withCtxt_preprocessed_flat.pickle',
        data_path_sequential='processed-datasets/L5_bins200ms_withCtxt_preprocessed.pickle',
        results_dir='results/baseline_temporal'
    )
    
    # Load data
    trainer._load_data()
    
    # Run corrected experiment
    results = trainer.run_baseline_experiment(
        train_split=0.8,     # 80% for training 
        epochs=200,          # Same as original
        lr=1e-3,            # Same as original
        batch_size=64,      # Same as original
        patience=20         # Same as original
    )
    
    print("\n" + "=" * 60)
    print("🏁 CORRECTED BASELINE EXPERIMENT COMPLETE")
    print("=" * 60)
    print("Results saved with temporal splitting (no data leakage).")
    print("Compare with original results to see performance drop.")
    print("Expected: LSTM R² drops from 0.98 → 0.75-0.85")


if __name__ == "__main__":
    main() 