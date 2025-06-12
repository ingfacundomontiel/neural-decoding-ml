import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, Any
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from datetime import datetime
import time

from models import create_baseline_models, get_model_summary
from utils.data_utils import create_data_loaders


class BaselineTrainer:
    """
    Comprehensive trainer for neural decoding baseline experiments.
    Handles MLP, RNN, and LSTM models with GPU support and full evaluation.
    """
    
    def __init__(self, 
                 data_path_flat: str,
                 data_path_sequential: str,
                 results_dir: str = "results/baseline",
                 device: str = "auto"):
        
        self.data_path_flat = data_path_flat
        self.data_path_sequential = data_path_sequential
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup with GPU support
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🔧 Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load data
        self._load_data()
        
        # Results storage
        self.results = {}
        
    def _load_data(self):
        """Load both flat and sequential data formats"""
        print("📂 Loading preprocessed data...")
        
        # Load flat data for MLP
        with open(self.data_path_flat, 'rb') as f:
            flat_data = pickle.load(f)
            self.X_flat, _, self.y_flat = flat_data
            
        # Load sequential data for RNN/LSTM  
        with open(self.data_path_sequential, 'rb') as f:
            seq_data = pickle.load(f)
            self.X_seq, _, self.y_seq = seq_data
            
        # Use the flat y for consistency (remove extra dimension if present)
        if self.y_flat.ndim > 1:
            self.y_flat = self.y_flat.squeeze()
        if self.y_seq.ndim > 1:
            self.y_seq = self.y_seq.squeeze()
            
        print(f"   Flat data: X{self.X_flat.shape}, y{self.y_flat.shape}")
        print(f"   Sequential data: X{self.X_seq.shape}, y{self.y_seq.shape}")
        print(f"   Position range: {np.min(self.y_flat):.1f} to {np.max(self.y_flat):.1f}")
        
    def _get_data_for_model(self, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get appropriate data format for model type"""
        if model_type == "MLP":
            return self.X_flat, self.y_flat
        else:  # RNN or LSTM
            return self.X_seq, self.y_seq
            
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        metrics = {
            'r2': r2_score(y_true_clean, y_pred_clean),
            'rmse': np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
            'mae': mean_absolute_error(y_true_clean, y_pred_clean),
            'correlation': pearsonr(y_true_clean, y_pred_clean)[0],
            'mean_relative_error': np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100
        }
        
        return metrics
        
    def train_model(self, 
                   model, 
                   train_loader, 
                   val_loader,
                   model_name: str,
                   epochs: int = 200,
                   lr: float = 1e-3,
                   patience: int = 20,
                   save_model: bool = True) -> Dict[str, Any]:
        """
        Train a single model with early stopping and comprehensive logging
        """
        print(f"\n🚀 Training {model_name}")
        print(f"   Parameters: {model.get_num_parameters():,}")
        print(f"   Architecture: {model.get_architecture_summary()}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Setup training components
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Training history
        train_losses = []
        val_losses = []
        val_r2_scores = []
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y.unsqueeze(1))
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    val_predictions.append(outputs.cpu().numpy())
                    val_targets.append(batch_y.cpu().numpy())
                    
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # Calculate validation R²
            val_pred_all = np.concatenate(val_predictions).squeeze()
            val_target_all = np.concatenate(val_targets)
            val_r2 = r2_score(val_target_all, val_pred_all)
            val_r2_scores.append(val_r2)
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1:3d}: Train Loss: {avg_train_loss:.4f}, "
                     f"Val Loss: {avg_val_loss:.4f}, Val R²: {val_r2:.4f}")
                
            # Early stopping
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
                
        training_time = time.time() - start_time
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        # Save model if requested
        if save_model:
            model_path = self.results_dir / f"{model_name}_best.pth"
            torch.save({
                'model_state_dict': best_model_state,
                'model_config': model.config,
                'training_history': {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_r2_scores': val_r2_scores
                }
            }, model_path)
            
        return {
            'model': model,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'epochs_trained': epoch + 1,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_r2_scores': val_r2_scores
            }
        }
        
    def evaluate_model(self, model, test_loader, model_name: str) -> Dict[str, float]:
        """Evaluate model on test set"""
        print(f"📊 Evaluating {model_name}")
        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
                
        # Combine all predictions and targets
        predictions = np.concatenate(all_predictions).squeeze()
        targets = np.concatenate(all_targets)
        
        # Calculate metrics
        metrics = self._calculate_metrics(targets, predictions)
        
        # Print results
        print(f"   R²: {metrics['r2']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.2f}")
        print(f"   MAE: {metrics['mae']:.2f}")
        print(f"   Correlation: {metrics['correlation']:.4f}")
        
        return metrics
        
    def run_baseline_experiment(self,
                              train_split: float = 0.8,
                              epochs: int = 200,
                              lr: float = 1e-3,
                              batch_size: int = 64,
                              patience: int = 20) -> Dict[str, Any]:
        """
        Run complete baseline experiment for all 9 models
        """
        print("🎯 STARTING BASELINE EXPERIMENT")
        print("=" * 60)
        print(f"Training parameters:")
        print(f"  Train/Test split: {train_split*100:.0f}%/{(1-train_split)*100:.0f}%")
        print(f"  Epochs: {epochs} (patience: {patience})")
        print(f"  Learning rate: {lr}")
        print(f"  Batch size: {batch_size}")
        
        all_results = {}
        
        # Get data dimensions for model creation
        num_features = self.X_seq.shape[2]  # 50 features
        sequence_length = self.X_seq.shape[1]  # 11 time bins
        
        # Create all baseline models
        models = create_baseline_models(num_features, sequence_length)
        
        # Train each model type
        for model_type in ['MLP', 'RNN', 'LSTM']:
            print(f"\n🔥 TRAINING {model_type} MODELS")
            print("-" * 40)
            
            # Get appropriate data format
            X, y = self._get_data_for_model(model_type)
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(
                X, y, model_type, batch_size=batch_size,
                validation_split=0.2, test_split=1-train_split,
                shuffle=True, random_seed=42
            )
            
            model_results = {}
            
            # Train all layer configurations
            for layer_config in ['1_layer', '2_layer', '3_layer']:
                model_name = f"{model_type}_{layer_config}"
                model = models[model_type][layer_config]
                
                # Train model
                train_result = self.train_model(
                    model, train_loader, val_loader, model_name,
                    epochs=epochs, lr=lr, patience=patience
                )
                
                # Evaluate on test set
                test_metrics = self.evaluate_model(
                    train_result['model'], test_loader, model_name
                )
                
                # Store results
                model_results[layer_config] = {
                    'model_info': train_result['model'].get_model_info(),
                    'training': {
                        'best_val_loss': train_result['best_val_loss'],
                        'training_time': train_result['training_time'],
                        'epochs_trained': train_result['epochs_trained']
                    },
                    'test_metrics': test_metrics
                }
                
            all_results[model_type] = model_results
            
        # Save comprehensive results
        self._save_results(all_results)
        
        # Generate comparison report
        self._generate_comparison_report(all_results)
        
        return all_results
        
    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON file"""
        # Convert any non-serializable objects
        serializable_results = {}
        for model_type, model_results in results.items():
            serializable_results[model_type] = {}
            for layer_config, result in model_results.items():
                serializable_results[model_type][layer_config] = {
                    'model_info': {
                        'model_type': result['model_info']['model_type'],
                        'num_parameters': result['model_info']['num_parameters'],
                        'expected_input_shape': result['model_info']['expected_input_shape']
                    },
                    'training': result['training'],
                    'test_metrics': result['test_metrics']
                }
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"baseline_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        print(f"\n💾 Results saved to: {results_file}")
        
    def _generate_comparison_report(self, results: Dict[str, Any]):
        """Generate a comprehensive comparison report"""
        print(f"\n" + "=" * 80)
        print("🏆 BASELINE EXPERIMENT RESULTS")
        print("=" * 80)
        
        # Create comparison table
        rows = []
        for model_type in ['MLP', 'RNN', 'LSTM']:
            for layer_config in ['1_layer', '2_layer', '3_layer']:
                result = results[model_type][layer_config]
                rows.append({
                    'Model': f"{model_type}",
                    'Layers': layer_config.replace('_layer', ''),
                    'Params': f"{result['model_info']['num_parameters']:,}",
                    'R²': f"{result['test_metrics']['r2']:.4f}",
                    'RMSE': f"{result['test_metrics']['rmse']:.2f}",
                    'MAE': f"{result['test_metrics']['mae']:.2f}",
                    'Corr': f"{result['test_metrics']['correlation']:.4f}",
                    'Time(s)': f"{result['training']['training_time']:.1f}"
                })
        
        # Print table
        headers = ['Model', 'Layers', 'Params', 'R²', 'RMSE', 'MAE', 'Corr', 'Time(s)']
        col_widths = [max(len(str(row[h])) for row in rows + [dict(zip(headers, headers))]) for h in headers]
        
        # Header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        
        # Rows
        for row in rows:
            row_line = " | ".join(str(row[h]).ljust(w) for h, w in zip(headers, col_widths))
            print(row_line)
            
        # Best model summary
        best_r2_model = max(rows, key=lambda x: float(x['R²']))
        best_rmse_model = min(rows, key=lambda x: float(x['RMSE']))
        
        print(f"\n🥇 BEST MODELS:")
        print(f"   Highest R²: {best_r2_model['Model']} ({best_r2_model['Layers']} layers) - R² = {best_r2_model['R²']}")
        print(f"   Lowest RMSE: {best_rmse_model['Model']} ({best_rmse_model['Layers']} layers) - RMSE = {best_rmse_model['RMSE']}")
        
        print(f"\n✅ Baseline experiment completed!")
        self.results = results 