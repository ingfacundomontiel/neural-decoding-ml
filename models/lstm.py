import torch
import torch.nn as nn
from .base_model import BaseNeuralDecoder
from .configs import LSTMConfig


class LSTM(BaseNeuralDecoder):
    """
    Long Short-Term Memory network for neural decoding.
    Supports multiple layers and bidirectional processing.
    """
    
    def __init__(self, config: LSTMConfig):
        super(LSTM, self).__init__(config)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout_rate if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(config.total_hidden_size, config.output_size)
        
        # Additional dropout for output if specified
        self.output_dropout = nn.Dropout(config.dropout_rate) if config.dropout_rate > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_size]
        
        Returns:
            torch.Tensor: Predicted position [batch_size, output_size]
        """
        # LSTM forward pass
        # output: [batch_size, sequence_length, hidden_size * num_directions]
        # (hidden, cell): ([num_layers * num_directions, batch_size, hidden_size], [...])
        output, (hidden, cell) = self.lstm(x)
        
        # Use the last time step output for prediction
        # If bidirectional, this includes both forward and backward hidden states
        last_output = output[:, -1, :]  # [batch_size, hidden_size * num_directions]
        
        # Apply output dropout if specified
        if self.output_dropout is not None:
            last_output = self.output_dropout(last_output)
        
        # Project to output size
        prediction = self.output_projection(last_output)  # [batch_size, output_size]
        
        return prediction
    
    @property
    def model_type(self) -> str:
        return "LSTM"
    
    def get_architecture_summary(self) -> str:
        """Return a human-readable summary of the architecture"""
        direction = "Bidirectional" if self.config.bidirectional else "Unidirectional"
        return (f"LSTM: {self.config.input_size} -> "
                f"{self.config.num_layers} layers x {self.config.hidden_size} hidden "
                f"({direction}) -> {self.config.output_size}") 