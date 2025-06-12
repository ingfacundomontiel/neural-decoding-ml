"""
Demo script showing how to use the flexible neural decoding model system.

This script demonstrates:
1. Creating baseline configurations for all model types
2. Instantiating models with different architectures
3. Comparing model sizes
4. Data handling for different model types
"""

import numpy as np
import torch
from models import (
    create_baseline_models, 
    get_model_summary, 
    compare_model_sizes,
    MLPConfig, 
    RNNConfig, 
    LSTMConfig,
    create_model
)
from utils.data_utils import get_data_info, prepare_data_for_model


def demo_baseline_models():
    """Demonstrate baseline model creation and comparison"""
    print("🚀 NEURAL DECODING MODEL SYSTEM DEMO")
    print("=" * 50)
    
    # Simulate your data dimensions (replace with actual values)
    num_samples = 1000
    num_time_bins = 11  # 5 before + 1 current + 5 after
    num_features = 50   # Number of neurons + context features
    
    # Create synthetic data for demo
    X = np.random.randn(num_samples, num_time_bins, num_features)
    y = np.random.randn(num_samples)  # Position values
    
    print(f"📊 Dataset Info:")
    data_info = get_data_info(X, y)
    for key, value in data_info.items():
        print(f"  {key}: {value}")
    
    print(f"\n🏗️ Creating Baseline Models...")
    
    # Create all baseline models (1, 2, 3 layers for each type)
    models = create_baseline_models(
        input_size=num_features, 
        sequence_length=num_time_bins
    )
    
    print(f"\n📈 Model Comparison:")
    print(compare_model_sizes(models))
    
    # Show detailed info for one model of each type
    print(f"\n🔍 Detailed Model Information:")
    print("\n" + "="*50)
    
    for model_type in ['MLP', 'RNN', 'LSTM']:
        model = models[model_type]['2_layer']  # Show 2-layer variant
        print(f"\n{model_type} (2-layer):")
        print("-" * 30)
        print(get_model_summary(model))


def demo_custom_configurations():
    """Demonstrate creating custom model configurations"""
    print(f"\n\n🛠️ CUSTOM MODEL CONFIGURATIONS")
    print("=" * 50)
    
    num_features = 50
    sequence_length = 11
    
    # Custom MLP with specific architecture
    custom_mlp_config = MLPConfig(
        input_size=num_features * sequence_length,
        hidden_sizes=[256, 128, 64],  # 3 layers with specific sizes
        dropout_rate=0.3,
        activation='relu'
    )
    
    # Custom RNN with GRU cells and bidirectional processing
    custom_rnn_config = RNNConfig(
        input_size=num_features,
        hidden_size=128,
        num_layers=2,
        rnn_type='GRU',
        bidirectional=True,
        dropout_rate=0.2,
        sequence_length=sequence_length
    )
    
    # Custom LSTM with bidirectional processing
    custom_lstm_config = LSTMConfig(
        input_size=num_features,
        hidden_size=256,
        num_layers=3,
        bidirectional=True,
        dropout_rate=0.25,
        sequence_length=sequence_length
    )
    
    # Create models from custom configs
    custom_mlp = create_model(custom_mlp_config)
    custom_rnn = create_model(custom_rnn_config)
    custom_lstm = create_model(custom_lstm_config)
    
    print("Custom Model Architectures:")
    print(f"MLP: {custom_mlp.get_architecture_summary()}")
    print(f"RNN: {custom_rnn.get_architecture_summary()}")
    print(f"LSTM: {custom_lstm.get_architecture_summary()}")
    
    print(f"\nParameter Counts:")
    print(f"Custom MLP: {custom_mlp.get_num_parameters():,} parameters")
    print(f"Custom RNN: {custom_rnn.get_num_parameters():,} parameters")
    print(f"Custom LSTM: {custom_lstm.get_num_parameters():,} parameters")


def demo_different_sequence_lengths():
    """Demonstrate how to handle different temporal window sizes"""
    print(f"\n\n⏱️ DIFFERENT TEMPORAL WINDOW SIZES")
    print("=" * 50)
    
    num_features = 50
    
    # Different temporal window configurations
    window_configs = [
        (5, 1, 5),    # Current: 5 before + 1 current + 5 after = 11 bins
        (10, 1, 10),  # Longer: 10 before + 1 current + 10 after = 21 bins  
        (3, 1, 3),    # Shorter: 3 before + 1 current + 3 after = 7 bins
    ]
    
    for before, current, after in window_configs:
        sequence_length = before + current + after
        print(f"\nTemporal Window: {before}-{current}-{after} (total: {sequence_length} bins)")
        
        # Create models for this configuration
        models = create_baseline_models(num_features, sequence_length)
        
        # Show parameter counts for 1-layer models
        print(f"  MLP 1-layer: {models['MLP']['1_layer'].get_num_parameters():,} params")
        print(f"  RNN 1-layer: {models['RNN']['1_layer'].get_num_parameters():,} params")
        print(f"  LSTM 1-layer: {models['LSTM']['1_layer'].get_num_parameters():,} params")


if __name__ == "__main__":
    # Run all demos
    demo_baseline_models()
    demo_custom_configurations()
    demo_different_sequence_lengths()
    
    print(f"\n\n✅ Demo completed!")
    print(f"\nNext steps:")
    print(f"1. Modify your preprocessing to output X with shape [samples, time_bins, features]")
    print(f"2. Use create_baseline_models() to get all your baseline models")
    print(f"3. Use the training infrastructure to train and compare models")
    print(f"4. For hyperparameter optimization, create custom configs and use create_model()") 