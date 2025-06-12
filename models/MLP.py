import torch
import torch.nn as nn
from .base_model import BaseNeuralDecoder
from .configs import MLPConfig


class MLP(BaseNeuralDecoder):
    """
    Multi-Layer Perceptron for neural decoding.
    Supports variable number of hidden layers with configurable sizes.
    """
    
    def __init__(self, config: MLPConfig):
        super(MLP, self).__init__(config)
        
        # Build layers dynamically based on config
        layers = []
        
        # Input layer
        prev_size = config.input_size
        
        # Hidden layers
        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Activation function
            if config.activation.lower() == 'relu':
                layers.append(nn.ReLU())
            elif config.activation.lower() == 'tanh':
                layers.append(nn.Tanh())
            elif config.activation.lower() == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                raise ValueError(f"Unsupported activation function: {config.activation}")
            
            # Dropout
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
                
            prev_size = hidden_size
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_size, config.output_size))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, flattened_features]
        
        Returns:
            torch.Tensor: Predicted position [batch_size, output_size]
        """
        return self.model(x)
    
    @property
    def model_type(self) -> str:
        return "MLP"
    
    def get_architecture_summary(self) -> str:
        """Return a human-readable summary of the architecture"""
        hidden_sizes_str = " -> ".join(map(str, self.config.hidden_sizes))
        return f"MLP: {self.config.input_size} -> {hidden_sizes_str} -> {self.config.output_size}" 