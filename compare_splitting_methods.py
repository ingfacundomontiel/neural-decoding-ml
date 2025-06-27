#!/usr/bin/env python3
"""
Compare time-bin based vs trial-based splitting for neural decoding baseline experiments.

This script runs both splitting methods and compares the results to demonstrate
the impact of data leakage on model evaluation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from baseline_trainer import BaselineTrainer


def run_comparison():
    """Run both splitting methods and compare results"""
    
    print("🔬 NEURAL DECODING SPLITTING METHOD COMPARISON")
    print("=" * 70)
    print("Comparing:")
    print("1. Time-bin based splitting (current baseline)")
    print("2. Trial-based splitting (no data leakage)")
    print("=" * 70)
    
    # Common parameters
    common_params = {
        'train_split': 0.8,
        'epochs': 50,  # Reduced for comparison
        'lr': 1e-3,
        'batch_size': 64,
        'patience': 10
    }
    
    results = {}
    
    # Run time-bin based experiment
    print("\n🔹 RUNNING TIME-BIN BASED EXPERIMENT")
    print("-" * 50)
    
    trainer_timebin = BaselineTrainer(
        data_path_flat="processed-datasets/L5_bins200ms_withCtxt_preprocessed_flat.pickle",
        data_path_sequential="processed-datasets/L5_bins200ms_withCtxt_preprocessed.pickle",
        results_dir="results/comparison_timebin",
        device="auto"
    )
    
    results['timebin'] = trainer_timebin.run_baseline_experiment(
        trial_based_split=False,
        **common_params
    )
    
    # Run trial-based experiment
    print("\n🔸 RUNNING TRIAL-BASED EXPERIMENT")
    print("-" * 50)
    
    trainer_trial = BaselineTrainer(
        data_path_flat="processed-datasets/L5_bins200ms_withCtxt_preprocessed_flat.pickle",
        data_path_sequential="processed-datasets/L5_bins200ms_withCtxt_preprocessed.pickle",
        results_dir="results/comparison_trial",
        device="auto"
    )
    
    results['trial'] = trainer_trial.run_baseline_experiment(
        trial_based_split=True,
        **common_params
    )
    
    # Compare results
    print("\n" + "=" * 100)
    print("📊 SPLITTING METHOD COMPARISON RESULTS")
    print("=" * 100)
    
    # Create comparison table
    comparison_data = []
    
    for model_type in ['MLP', 'RNN', 'LSTM']:
        for layer_config in ['1_layer', '2_layer', '3_layer']:
            timebin_result = results['timebin'][model_type][layer_config]['test_metrics']
            trial_result = results['trial'][model_type][layer_config]['test_metrics']
            
            comparison_data.append({
                'Model': f"{model_type}_{layer_config}",
                'TimeB_R2': timebin_result['r2'],
                'Trial_R2': trial_result['r2'],
                'R2_Diff': timebin_result['r2'] - trial_result['r2'],
                'TimeB_RMSE': timebin_result['rmse'],
                'Trial_RMSE': trial_result['rmse'],
                'RMSE_Diff': trial_result['rmse'] - timebin_result['rmse'],
                'TimeB_Corr': timebin_result['correlation'],
                'Trial_Corr': trial_result['correlation'],
                'Corr_Diff': timebin_result['correlation'] - trial_result['correlation']
            })
    
    # Create DataFrame for easy viewing
    df = pd.DataFrame(comparison_data)
    
    print("\nDetailed Comparison (Time-bin vs Trial-based):")
    print("=" * 120)
    
    # Print formatted table
    headers = ['Model', 'R² (TimeB)', 'R² (Trial)', 'R² Diff', 'RMSE (TimeB)', 'RMSE (Trial)', 'RMSE Diff', 'Corr (TimeB)', 'Corr (Trial)', 'Corr Diff']
    col_widths = [12, 10, 10, 8, 11, 11, 9, 11, 11, 9]
    
    # Header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    
    # Rows
    for _, row in df.iterrows():
        row_values = [
            row['Model'],
            f"{row['TimeB_R2']:.4f}",
            f"{row['Trial_R2']:.4f}",
            f"{row['R2_Diff']:.4f}",
            f"{row['TimeB_RMSE']:.1f}",
            f"{row['Trial_RMSE']:.1f}",
            f"{row['RMSE_Diff']:.1f}",
            f"{row['TimeB_Corr']:.4f}",
            f"{row['Trial_Corr']:.4f}",
            f"{row['Corr_Diff']:.4f}"
        ]
        row_line = " | ".join(val.ljust(w) for val, w in zip(row_values, col_widths))
        print(row_line)
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print(f"Average R² difference (TimeB - Trial): {df['R2_Diff'].mean():.4f}")
    print(f"Average RMSE difference (Trial - TimeB): {df['RMSE_Diff'].mean():.1f}")
    print(f"Average Correlation difference (TimeB - Trial): {df['Corr_Diff'].mean():.4f}")
    
    # Data leakage analysis
    positive_r2_diff = (df['R2_Diff'] > 0).sum()
    total_models = len(df)
    
    print(f"\n🚨 DATA LEAKAGE ANALYSIS:")
    print(f"Models with higher R² in time-bin splitting: {positive_r2_diff}/{total_models} ({100*positive_r2_diff/total_models:.1f}%)")
    print(f"This suggests data leakage in time-bin based splitting!")
    
    # Best models comparison
    print(f"\n🏆 BEST MODEL COMPARISON:")
    
    timebin_best = df.loc[df['TimeB_R2'].idxmax()]
    trial_best = df.loc[df['Trial_R2'].idxmax()]
    
    print(f"Best time-bin model: {timebin_best['Model']} (R² = {timebin_best['TimeB_R2']:.4f})")
    print(f"Best trial-based model: {trial_best['Model']} (R² = {trial_best['Trial_R2']:.4f})")
    
    # Trial split information
    if hasattr(trainer_trial, 'trial_split_info'):
        split_info = trainer_trial.trial_split_info
        print(f"\n🎯 TRIAL SPLIT INFORMATION:")
        print(f"Train trials: {split_info['train_trials']} ({split_info['train_samples']} samples)")
        print(f"Val trials:   {split_info['val_trials']} ({split_info['val_samples']} samples)")
        print(f"Test trials:  {split_info['test_trials']} ({split_info['test_samples']} samples)")
        
        total_samples = split_info['train_samples'] + split_info['val_samples'] + split_info['test_samples']
        print(f"Total samples: {total_samples}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"✅ Use trial-based splitting for:")
    print(f"   - Realistic model evaluation")
    print(f"   - Avoiding data leakage")
    print(f"   - Better generalization assessment")
    print(f"   - More conservative performance estimates")
    print(f"")
    print(f"⚠️  Time-bin splitting may be inflating performance due to:")
    print(f"   - Temporal correlations between train/test")
    print(f"   - Information leakage within trials")
    print(f"   - Overly optimistic R² values")
    
    return results


if __name__ == "__main__":
    try:
        import pandas as pd
        results = run_comparison()
    except ImportError:
        print("❌ pandas is required for this comparison script.")
        print("   Install with: pip install pandas")
        print("   Or run the experiments separately:")
        print("   python3 run_baseline.py  # time-bin based")
        print("   python3 run_baseline_trial_based.py  # trial-based") 