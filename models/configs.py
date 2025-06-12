from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MLPConfig:
    """Configuration for Multi-Layer Perceptron model"""
    input_size: int
    hidden_sizes: List[int]  # List of hidden layer sizes, e.g., [128] for 1 layer, [128, 64] for 2 layers
    output_size: int = 1
    dropout_rate: float = 0.2
    activation: str = 'relu'  # 'relu', 'tanh', 'sigmoid'
    
    @property
    def num_layers(self) -> int:
        return len(self.hidden_sizes)


@dataclass 
class RNNConfig:
    """Configuration for RNN model"""
    input_size: int
    hidden_size: int
    num_layers: int = 1
    output_size: int = 1
    rnn_type: str = 'RNN'  # 'RNN' or 'GRU'
    dropout_rate: float = 0.2
    bidirectional: bool = False
    sequence_length: int = 11  # Default 5-before + 1-current + 5-after
    
    @property
    def total_hidden_size(self) -> int:
        """Hidden size accounting for bidirectional"""
        return self.hidden_size * (2 if self.bidirectional else 1)


@dataclass
class LSTMConfig:
    """Configuration for LSTM model"""
    input_size: int
    hidden_size: int
    num_layers: int = 1
    output_size: int = 1
    dropout_rate: float = 0.2
    bidirectional: bool = False
    sequence_length: int = 11  # Default 5-before + 1-current + 5-after
    
    @property
    def total_hidden_size(self) -> int:
        """Hidden size accounting for bidirectional"""
        return self.hidden_size * (2 if self.bidirectional else 1)


# Baseline configurations for quick experimentation
def get_baseline_configs(input_size: int, sequence_length: int = 11) -> dict:
    """
    Get baseline configurations for all models with 1, 2, and 3 layer variants
    
    Args:
        input_size: Number of input features (neurons)
        sequence_length: Length of temporal sequence for RNN/LSTM
    """
    configs = {}
    
    # MLP baselines - flattened input size is input_size * sequence_length
    flattened_input = input_size * sequence_length
    configs['MLP'] = {
        '1_layer': MLPConfig(input_size=flattened_input, hidden_sizes=[128]),
        '2_layer': MLPConfig(input_size=flattened_input, hidden_sizes=[128, 64]),
        '3_layer': MLPConfig(input_size=flattened_input, hidden_sizes=[128, 64, 32])
    }
    
    # RNN baselines
    configs['RNN'] = {
        '1_layer': RNNConfig(input_size=input_size, hidden_size=64, num_layers=1, sequence_length=sequence_length),
        '2_layer': RNNConfig(input_size=input_size, hidden_size=64, num_layers=2, sequence_length=sequence_length),
        '3_layer': RNNConfig(input_size=input_size, hidden_size=64, num_layers=3, sequence_length=sequence_length)
    }
    
    # LSTM baselines  
    configs['LSTM'] = {
        '1_layer': LSTMConfig(input_size=input_size, hidden_size=64, num_layers=1, sequence_length=sequence_length),
        '2_layer': LSTMConfig(input_size=input_size, hidden_size=64, num_layers=2, sequence_length=sequence_length),
        '3_layer': LSTMConfig(input_size=input_size, hidden_size=64, num_layers=3, sequence_length=sequence_length)
    }
    
    return configs 