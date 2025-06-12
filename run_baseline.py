#!/usr/bin/env python3
"""
Run baseline experiment for neural decoding project.

This script trains and evaluates all 9 baseline models:
- MLP: 1, 2, 3 layers
- RNN: 1, 2, 3 layers  
- LSTM: 1, 2, 3 layers

With 80/20 train/test split and comprehensive evaluation metrics.
"""

from baseline_trainer import BaselineTrainer


def main():
    """Run the complete baseline experiment"""
    
    print("🧠 NEURAL DECODING BASELINE EXPERIMENT")
    print("=" * 50)
    print("Training 9 models (MLP, RNN, LSTM × 1,2,3 layers)")
    print("Data: L5 dataset with 200ms time bins")
    print("Split: 80% train, 20% test")
    print("=" * 50)
    
    # Initialize trainer with your data paths
    trainer = BaselineTrainer(
        data_path_flat="processed-datasets/L5_bins200ms_withCtxt_preprocessed_flat.pickle",
        data_path_sequential="processed-datasets/L5_bins200ms_withCtxt_preprocessed.pickle",
        results_dir="results/baseline",
        device="auto"  # Automatically use GPU if available
    )
    
    # Run the complete baseline experiment
    results = trainer.run_baseline_experiment(
        train_split=0.8,      # 80% for training
        epochs=200,           # Maximum epochs
        lr=1e-3,             # Learning rate (Adam optimizer)
        batch_size=64,        # Batch size
        patience=20          # Early stopping patience
    )
    
    print(f"\n🎉 Experiment completed successfully!")
    print(f"📁 Results saved in: results/baseline/")
    print(f"📊 Models trained: {sum(len(models) for models in results.values())}")
    
    # Quick summary of best performers
    print(f"\n📈 Quick Summary:")
    for model_type in ['MLP', 'RNN', 'LSTM']:
        best_config = max(results[model_type].items(), 
                         key=lambda x: x[1]['test_metrics']['r2'])
        config_name, result = best_config
        print(f"   Best {model_type}: {config_name} (R² = {result['test_metrics']['r2']:.4f})")


if __name__ == "__main__":
    main() 