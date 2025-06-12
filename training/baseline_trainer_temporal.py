#!/usr/bin/env python3
"""
Temporal Baseline Trainer - Fixed version with proper data splitting

This trainer fixes the critical data leakage issue by using temporal/contiguous
data splitting instead of random splitting for time series data.
"""

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr

# Import corrected data utilities
import sys
sys.path.append('.')
from utils.temporal_data_utils import create_temporal_data_loaders
from models.model_factory import create_baseline_models


class BaselineTrainerTemporal:
    """
    Temporal Baseline Trainer with proper data splitting
    
    Key difference from original: Uses temporal/contiguous data splitting
    to avoid data leakage in time series with temporal windows.
    """
    
    def __init__(self, data_path_flat: str, data_path_sequential: str, results_dir: str = "results/baseline_temporal"):
        """
        Initialize the temporal trainer
        
        Args:
            data_path_flat: Path to flattened data for MLP
            data_path_sequential: Path to sequential data for RNN/LSTM
            results_dir: Directory to save results
        """
        self.data_path_flat = data_path_flat
        self.data_path_sequential = data_path_sequential
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # GPU setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔥 Using device: {self.device}")
        
        # Data containers
        self.X_flat = None
        self.y_flat = None
        self.X_seq = None
        self.y_seq = None
        
        # Experiment tracking
        self.experiment_start_time = None
        self.experiment_params = None
        self.experiment_dir = None
        
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
        print(f"   ✅ TEMPORAL SPLITTING: Will use contiguous splits (no leakage)")
        
    def _get_data_for_model(self, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get appropriate data format for model type"""
        if model_type == "MLP":
            return self.X_flat, self.y_flat
        else:  # RNN or LSTM
            return self.X_seq, self.y_seq
            
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
        Train a single model with early stopping and detailed metrics
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            model_name: Name for saving/logging
            epochs: Maximum epochs
            lr: Learning rate
            patience: Early stopping patience
            save_model: Whether to save the best model
            
        Returns:
            Dict with training results and metrics
        """
        print(f"   🎯 Training {model_name}...")
        
        # Move model to device
        model = model.to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Tracking variables
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_r2_scores = []
        best_model_state = None
        start_time = datetime.now()
        
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
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            # Print progress periodically
            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"      Epoch {epoch+1:3d}: Train Loss={avg_train_loss:.4f}, "
                      f"Val Loss={avg_val_loss:.4f}, Val R²={val_r2:.4f}")
                
            # Early stopping
            if patience_counter >= patience:
                print(f"      Early stopping at epoch {epoch+1} (patience={patience})")
                break
                
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        training_time = datetime.now() - start_time
        epochs_trained = epoch + 1
        
        # Save model if requested
        if save_model:
            model_path = self.experiment_dir / f"{model_name}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_info': model.get_model_info(),
                'training_history': {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_r2_scores': val_r2_scores
                }
            }, model_path)
            
        print(f"      ✅ Completed in {training_time.total_seconds():.1f}s "
              f"({epochs_trained} epochs)")
        
        return {
            'model': model,
            'best_val_loss': best_val_loss,
            'training_time': training_time.total_seconds(),
            'epochs_trained': epochs_trained,
            'training_history': {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_r2_scores': val_r2_scores
            }
        }
        
    def evaluate_model(self, model, test_loader, model_name: str) -> Dict[str, float]:
        """
        Evaluate model on test set with comprehensive metrics
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            model_name: Name for logging
            
        Returns:
            Dict with test metrics
        """
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = model(batch_X)
                predictions.append(outputs.cpu().numpy())
                targets.append(batch_y.cpu().numpy())
                
        # Concatenate all predictions and targets
        y_pred = np.concatenate(predictions).squeeze()
        y_true = np.concatenate(targets)
        
        # Calculate comprehensive metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        correlation, p_value = pearsonr(y_true, y_pred)
        
        # Calculate normalized metrics
        y_range = np.max(y_true) - np.min(y_true)
        normalized_rmse = rmse / y_range
        normalized_mae = mae / y_range
        
        metrics = {
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(correlation),
            'correlation_p_value': float(p_value),
            'normalized_rmse': float(normalized_rmse),
            'normalized_mae': float(normalized_mae),
            'position_range': float(y_range)
        }
        
        print(f"      📊 {model_name} Test Results:")
        print(f"         R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
        print(f"         Correlation = {correlation:.4f}, Norm. RMSE = {normalized_rmse:.4f}")
        
        return metrics
        
    def run_baseline_experiment(self,
                              train_split: float = 0.8,
                              epochs: int = 200,
                              lr: float = 1e-3,
                              batch_size: int = 64,
                              patience: int = 20) -> Dict[str, Any]:
        """
        Run complete baseline experiment with TEMPORAL splitting (corrected)
        """
        # Track experiment start time and parameters
        self.experiment_start_time = datetime.now()
        self.experiment_params = {
            'train_split': train_split,
            'epochs': epochs,
            'lr': lr,
            'batch_size': batch_size,
            'patience': patience,
            'splitting_method': 'TEMPORAL_CONTIGUOUS'  # Key difference!
        }
        
        # Create timestamped experiment directory
        timestamp = self.experiment_start_time.strftime("%d-%m-%Y--%H-%M-%S")
        self.experiment_dir = self.results_dir / f"temporal_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        print("🚨 CORRECTED BASELINE EXPERIMENT (TEMPORAL SPLITTING)")
        print("=" * 60)
        print(f"Experiment Directory: {self.experiment_dir}")
        print(f"Start Time: {self.experiment_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training parameters:")
        print(f"  Train/Test split: {train_split*100:.0f}%/{(1-train_split)*100:.0f}%")
        print(f"  Splitting method: TEMPORAL/CONTIGUOUS (fixes data leakage)")
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
            print(f"\n🔥 TRAINING {model_type} MODELS (TEMPORAL SPLITTING)")
            print("-" * 50)
            
            # Get appropriate data format
            X, y = self._get_data_for_model(model_type)
            
            # 🚨 KEY FIX: Use temporal data loaders instead of random
            train_loader, val_loader, test_loader = create_temporal_data_loaders(
                X, y, model_type, batch_size=batch_size,
                validation_split=0.2, test_split=1-train_split,
                random_seed=42
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
        
        # Calculate total experiment time
        total_time = datetime.now() - self.experiment_start_time
        
        print(f"\n🏁 TEMPORAL BASELINE EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Total time: {total_time}")
        print(f"Results saved in: {self.experiment_dir}")
        print(f"Splitting method: TEMPORAL (no data leakage)")
        
        return all_results
        
    def _save_results(self, results: Dict[str, Any]):
        """Save comprehensive experiment results"""
        
        # Save complete results as pickle
        results_path = self.experiment_dir / "complete_results.pickle"
        with open(results_path, 'wb') as f:
            pickle.dump({
                'results': results,
                'experiment_params': self.experiment_params,
                'experiment_time': self.experiment_start_time,
                'splitting_method': 'TEMPORAL_CONTIGUOUS'
            }, f)
            
        # Save summary as text
        summary_path = self.experiment_dir / "experiment_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("CORRECTED BASELINE EXPERIMENT RESULTS (TEMPORAL SPLITTING)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Experiment Time: {self.experiment_start_time}\n")
            f.write(f"Splitting Method: TEMPORAL/CONTIGUOUS (fixes data leakage)\n")
            f.write(f"Parameters: {self.experiment_params}\n\n")
            
            f.write("PERFORMANCE SUMMARY:\n")
            f.write("-" * 20 + "\n")
            
            for model_type in ['MLP', 'RNN', 'LSTM']:
                f.write(f"\n{model_type} MODELS:\n")
                for layer_config in ['1_layer', '2_layer', '3_layer']:
                    if layer_config in results[model_type]:
                        metrics = results[model_type][layer_config]['test_metrics']
                        r2 = metrics['r2_score']
                        rmse = metrics['rmse']
                        f.write(f"  {layer_config}: R² = {r2:.4f}, RMSE = {rmse:.2f}\n")
                    else:
                        f.write(f"  {layer_config}: FAILED\n")
                        
        print(f"Results saved to: {results_path}")
        print(f"Summary saved to: {summary_path}") 