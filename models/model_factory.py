from typing import Union
from .configs import MLPConfig, RNNConfig, LSTMConfig, get_baseline_configs
from .MLP import MLP
from .rnn import RNN
from .lstm import LSTM
from .base_model import BaseNeuralDecoder


def create_model(config: Union[MLPConfig, RNNConfig, LSTMConfig]) -> BaseNeuralDecoder:
    """
    Factory function to create models from configurations
    
    Args:
        config: Model configuration (MLPConfig, RNNConfig, or LSTMConfig)
    
    Returns:
        BaseNeuralDecoder: Instantiated model
    """
    if isinstance(config, MLPConfig):
        return MLP(config)
    elif isinstance(config, RNNConfig):
        return RNN(config)
    elif isinstance(config, LSTMConfig):
        return LSTM(config)
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")


def create_baseline_models(input_size: int, sequence_length: int = 11) -> dict:
    """
    Create all baseline models (1, 2, 3 layers for each type)
    
    Args:
        input_size: Number of input features (neurons)
        sequence_length: Length of temporal sequence for RNN/LSTM
    
    Returns:
        dict: Nested dictionary with model instances
              Format: {model_type: {layer_config: model_instance}}
    """
    configs = get_baseline_configs(input_size, sequence_length)
    models = {}
    
    for model_type, layer_configs in configs.items():
        models[model_type] = {}
        for layer_name, config in layer_configs.items():
            models[model_type][layer_name] = create_model(config)
    
    return models


def get_model_summary(model: BaseNeuralDecoder) -> str:
    """
    Get a comprehensive summary of a model
    
    Args:
        model: Model instance
    
    Returns:
        str: Formatted model summary
    """
    info = model.get_model_info()
    summary = f"""
Model Type: {info['model_type']}
Architecture: {model.get_architecture_summary()}
Parameters: {info['num_parameters']:,}
Expected Input Shape: {info['expected_input_shape']}
"""
    return summary.strip()


def compare_model_sizes(models: dict) -> str:
    """
    Compare parameter counts across different models
    
    Args:
        models: Dictionary of models (output from create_baseline_models)
    
    Returns:
        str: Formatted comparison table
    """
    comparison = "Model Comparison:\n"
    comparison += "=" * 50 + "\n"
    
    for model_type, layer_configs in models.items():
        comparison += f"\n{model_type}:\n"
        comparison += "-" * 20 + "\n"
        
        for layer_name, model in layer_configs.items():
            num_params = model.get_num_parameters()
            comparison += f"  {layer_name}: {num_params:,} parameters\n"
    
    return comparison 