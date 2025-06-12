import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Union
from .configs import MLPConfig, RNNConfig, LSTMConfig


class BaseNeuralDecoder(nn.Module, ABC):
    """
    Abstract base class for all neural decoding models.
    Ensures consistent interface across MLP, RNN, and LSTM implementations.
    """
    
    def __init__(self, config: Union[MLPConfig, RNNConfig, LSTMConfig]):
        super(BaseNeuralDecoder, self).__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model
        
        Args:
            x: Input tensor with appropriate shape for model type
               - MLP: [batch_size, flattened_features]
               - RNN/LSTM: [batch_size, sequence_length, features]
        
        Returns:
            torch.Tensor: Predicted position values [batch_size, 1]
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier"""
        pass
    
    @property
    def input_shape_info(self) -> str:
        """Return expected input shape information"""
        if hasattr(self.config, 'sequence_length'):
            return f"[batch_size, {self.config.sequence_length}, {self.config.input_size}]"
        else:
            return f"[batch_size, {self.config.input_size}]"
    
    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """Return comprehensive model information"""
        return {
            'model_type': self.model_type,
            'config': self.config,
            'num_parameters': self.get_num_parameters(),
            'expected_input_shape': self.input_shape_info
        } 