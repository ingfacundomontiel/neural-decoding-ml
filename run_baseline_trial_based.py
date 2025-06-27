#!/usr/bin/env python3
"""
Run trial-based baseline experiment for neural decoding project.

This script trains and evaluates all 9 baseline models using TRIAL-BASED data splitting:
- MLP: 1, 2, 3 layers
- RNN: 1, 2, 3 layers  
- LSTM: 1, 2, 3 layers

Key difference from run_baseline.py:
- Splits data by complete TRIALS instead of individual time bins
- Ensures no data leakage between train/test sets
- Provides more realistic evaluation of model generalization

With 80/20 train/test split (by trials) and comprehensive evaluation metrics.
"""

from baseline_trainer import BaselineTrainer


def main():
    """Run the complete trial-based baseline experiment"""
    
    print("🧠 NEURAL DECODING TRIAL-BASED BASELINE EXPERIMENT")
    print("=" * 60)
    print("Training 9 models (MLP, RNN, LSTM × 1,2,3 layers)")
    print("Data: L5 dataset with 200ms time bins")
    print("Split: 80% trials for train, 20% trials for test")
    print("Method: TRIAL-BASED splitting (no data leakage)")
    print("=" * 60)
    
    # Initialize trainer with your data paths
    trainer = BaselineTrainer(
        data_path_flat="processed-datasets/L5_bins200ms_withCtxt_preprocessed_flat.pickle",
        data_path_sequential="processed-datasets/L5_bins200ms_withCtxt_preprocessed.pickle",
        results_dir="results/baseline_trial_based",
        device="auto"  # Automatically use GPU if available
    )
    
    # Check if trial-based splitting is possible
    if trainer.trial_ids_flat is None:
        print("❌ ERROR: Trial IDs not found in data files!")
        print("   Trial-based splitting requires trial_ids to be saved with the data.")
        print("   Please re-run the preprocessing script to include trial_ids in the saved files.")
        return
    
    # Run the complete baseline experiment with trial-based splitting
    results = trainer.run_baseline_experiment(
        train_split=0.8,        # 80% of trials for training
        epochs=200,             # Maximum epochs
        lr=1e-3,               # Learning rate (Adam optimizer)
        batch_size=64,          # Batch size
        patience=20,            # Early stopping patience
        trial_based_split=True  # Use trial-based splitting
    )
    
    print(f"\n🎉 Trial-based experiment completed successfully!")
    print(f"📁 Results saved in: results/baseline_trial_based/")
    print(f"📊 Models trained: {sum(len(models) for models in results.values())}")
    
    # Quick summary of best performers
    print(f"\n📈 Quick Summary:")
    for model_type in ['MLP', 'RNN', 'LSTM']:
        best_config = max(results[model_type].items(), 
                         key=lambda x: x[1]['test_metrics']['r2'])
        config_name, result = best_config
        print(f"   Best {model_type}: {config_name} (R² = {result['test_metrics']['r2']:.4f})")
    
    # Trial split information
    if hasattr(trainer, 'trial_split_info'):
        split_info = trainer.trial_split_info
        print(f"\n🎯 Trial Split Information:")
        print(f"   Train trials: {split_info['train_trials']} ({split_info['train_samples']} samples)")
        print(f"   Val trials:   {split_info['val_trials']} ({split_info['val_samples']} samples)")
        print(f"   Test trials:  {split_info['test_trials']} ({split_info['test_samples']} samples)")


if __name__ == "__main__":
    main() 