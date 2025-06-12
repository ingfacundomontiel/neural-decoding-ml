#!/usr/bin/env python3
"""
Analyze potential data leakage in baseline experiment results.

This script demonstrates why the high LSTM performance (R² = 0.980) 
might be due to temporal data leakage from random splitting.
"""

import pickle
import numpy as np
from utils.temporal_data_utils import compare_random_vs_temporal_split, check_temporal_leakage


def analyze_current_results():
    """Analyze the data leakage issue in current baseline results"""
    
    print("🚨 DATA LEAKAGE ANALYSIS FOR BASELINE RESULTS")
    print("=" * 60)
    print("Investigating why LSTM achieved R² = 0.980 (suspiciously high)")
    print()
    
    # Load the data that was used in baseline experiment
    print("📂 Loading L5 dataset...")
    with open('processed-datasets/L5_bins200ms_withCtxt_preprocessed.pickle', 'rb') as f:
        seq_data = pickle.load(f)
        X_seq, _, y_seq = seq_data
        
    if y_seq.ndim > 1:
        y_seq = y_seq.squeeze()
        
    print(f"   Data shape: X{X_seq.shape}, y{y_seq.shape}")
    print(f"   Position range: {np.min(y_seq):.1f} to {np.max(y_seq):.1f}")
    print(f"   Temporal windows: 11 bins (5 before + 1 current + 5 after)")
    print()
    
    # Analyze splitting strategies
    print("🔍 PROBLEM ANALYSIS:")
    print("-" * 30)
    
    # Demonstrate the leakage issue
    leakage_results = compare_random_vs_temporal_split(X_seq, y_seq, test_split=0.2)
    
    print(f"\n💥 LEAKAGE IMPACT:")
    print(f"   Random split leakage: {leakage_results['random_leakage_count']} test samples")
    print(f"   Percentage affected: {leakage_results['random_leakage_count']/len(y_seq)*5:.1f}% of test set")
    print(f"   Risk level: {leakage_results['temporal_leakage_analysis']['leakage_risk']}")
    
    return leakage_results


def explain_why_results_too_good():
    """Explain why the baseline results are likely inflated"""
    
    print(f"\n🎯 WHY R² = 0.980 IS SUSPICIOUS:")
    print("=" * 40)
    
    print("📊 TYPICAL NEURAL DECODING PERFORMANCE:")
    print("   • Motor cortex → movement: R² = 0.7-0.9")
    print("   • Visual cortex → images: R² = 0.6-0.8") 
    print("   • Hippocampus → position: R² = 0.8-0.9")
    print("   • Piriform cortex → position: R² = ??? (our task)")
    print()
    print("❌ OUR RESULTS (with leakage):")
    print("   • LSTM 2-layer: R² = 0.980 (almost perfect!)")
    print("   • LSTM 1-layer: R² = 0.965 (also very high)")
    print("   • RNN 1-layer: R² = 0.886 (more reasonable)")
    print("   • MLP models: R² ≈ 0.56 (expected level)")
    print()
    print("🚨 RED FLAGS:")
    print("   1. Performance too close to perfect (98%)")
    print("   2. Huge gap between LSTM and MLP (75% difference)")
    print("   3. Multi-layer models failing (training instability)")
    print("   4. Random data splitting with temporal windows")
    print()
    print("🎯 LIKELY EXPLANATION:")
    print("   The LSTM is 'cheating' by seeing future information")
    print("   through overlapping temporal windows in the training set.")


def predict_corrected_performance():
    """Predict what performance should be with proper temporal splitting"""
    
    print(f"\n🔮 EXPECTED PERFORMANCE WITH PROPER SPLITTING:")
    print("=" * 50)
    
    print("📉 LIKELY PERFORMANCE DROPS:")
    print("   • LSTM 2-layer: 0.980 → 0.75-0.85 (realistic range)")
    print("   • LSTM 1-layer: 0.965 → 0.70-0.80")
    print("   • RNN 1-layer: 0.886 → 0.65-0.75 (less affected)")
    print("   • MLP models: ~0.56 → ~0.55 (minimal change)")
    print()
    print("🎯 WHY THESE PREDICTIONS:")
    print("   • LSTM benefits most from temporal patterns")
    print("   • With leakage, it sees 'future' neural states")
    print("   • Proper splitting forces true prediction")
    print("   • MLP less affected (already spatial pattern)")
    print()
    print("✅ WHAT GOOD RESULTS WOULD LOOK LIKE:")
    print("   • LSTM: R² = 0.75-0.85 (still best)")
    print("   • RNN: R² = 0.65-0.75 (competitive)")
    print("   • MLP: R² = 0.55-0.65 (reasonable baseline)")
    print("   • Smaller gaps between architectures")
    print("   • More stable training across layer counts")


def recommend_next_steps():
    """Recommend immediate actions to fix the issues"""
    
    print(f"\n🛠️ IMMEDIATE ACTION PLAN:")
    print("=" * 30)
    
    print("1. 🚨 CRITICAL: Re-run baseline with temporal splitting")
    print("   • Use utils.temporal_data_utils.create_temporal_data_loaders()")
    print("   • Compare old vs new results")
    print("   • Document performance drop")
    print()
    print("2. 🔍 ANALYZE: Investigate training stability")
    print("   • Add gradient clipping for multi-layer models")
    print("   • Try different learning rates")
    print("   • Monitor gradient norms during training")
    print()
    print("3. 📊 VALIDATE: Cross-check with literature")
    print("   • Research piriform cortex spatial coding")
    print("   • Compare with similar neural decoding studies")
    print("   • Assess if 75-85% R² is reasonable")
    print()
    print("4. 📈 IMPROVE: Optimize legitimate approaches")
    print("   • Hyperparameter search on properly split data")
    print("   • Try regularization techniques")
    print("   • Experiment with different architectures")
    print()
    print("🎯 NEW BASELINE TARGET:")
    print("   Achieve R² = 0.75-0.85 with proper temporal splitting")
    print("   This would be excellent and publishable results!")


def main():
    """Run complete data leakage analysis"""
    
    # Run analysis
    leakage_results = analyze_current_results()
    explain_why_results_too_good()
    predict_corrected_performance()
    recommend_next_steps()
    
    print(f"\n" + "=" * 60)
    print("🏁 CONCLUSION: BASELINE RESULTS LIKELY INVALID")
    print("=" * 60)
    print("❌ Current R² = 0.980 is likely due to data leakage")
    print("✅ Need to re-run with proper temporal splitting")
    print("🎯 Target: R² = 0.75-0.85 would be excellent results")
    print("📚 This is a common mistake in time series ML!")
    print()
    print("Next step: Run 'python run_baseline_temporal.py'")


if __name__ == "__main__":
    main() 