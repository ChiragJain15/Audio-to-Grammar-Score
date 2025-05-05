import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_squared_error
from data_processor import get_data_loaders, get_test_loader
from model import GrammarScoringModel
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_metrics(train_metrics, val_metrics, fold, output_dir):
    """Plot training and validation metrics"""
    metrics = ['loss', 'rmse', 'mae']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.plot(train_metrics[metric], label='Train')
        ax.plot(val_metrics[metric], label='Validation')
        ax.set_title(f'{metric.upper()} vs Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fold_{fold}_metrics.png'))
    plt.close()

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(output.detach().cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = np.mean(np.abs(np.array(all_targets) - np.array(all_preds)))
    
    return avg_loss, rmse, mae

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            all_preds.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = np.mean(np.abs(np.array(all_targets) - np.array(all_preds)))
    
    return avg_loss, rmse, mae

def train_fold(fold, train_loader, val_loader, device):
    """Train model for one fold"""
    model = GrammarScoringModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training loop
    best_val_rmse = float('inf')
    best_model = None
    train_metrics = {'loss': [], 'rmse': [], 'mae': []}
    val_metrics = {'loss': [], 'rmse': [], 'mae': []}
    
    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss, train_rmse, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_rmse, val_mae = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Log metrics
        train_metrics['loss'].append(train_loss)
        train_metrics['rmse'].append(train_rmse)
        train_metrics['mae'].append(train_mae)
        val_metrics['loss'].append(val_loss)
        val_metrics['rmse'].append(val_rmse)
        val_metrics['mae'].append(val_mae)
        
        logger.info(
            f"Fold {fold}, Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, Train MAE: {train_mae:.4f} - "
            f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}"
        )
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model = model.state_dict().copy()
            torch.save(
                best_model,
                os.path.join(Config.OUTPUT_DIR, "models", f"best_model_fold_{fold}.pt")
            )
    
    # Plot metrics
    plot_metrics(train_metrics, val_metrics, fold, os.path.join(Config.OUTPUT_DIR, "plots"))
    
    # Load best model
    model.load_state_dict(best_model)
    return model, best_val_rmse

def main():
    """Main training function"""
    # Set random seed
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # Get device
    device = Config.DEVICE
    logger.info(f"Using device: {device}")
    
    # Get data loaders
    train_loaders, val_loaders = get_data_loaders()
    
    # Train models for each fold
    models = []
    fold_rmse = []
    
    for fold, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        logger.info(f"\nTraining Fold {fold + 1}/{Config.N_FOLDS}")
        model, best_val_rmse = train_fold(fold + 1, train_loader, val_loader, device)
        models.append(model)
        fold_rmse.append(best_val_rmse)
    
    # Log cross-validation results
    logger.info("\nCross-validation results:")
    for fold, rmse in enumerate(fold_rmse):
        logger.info(f"Fold {fold + 1}: RMSE = {rmse:.4f}")
    logger.info(f"Mean RMSE: {np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f}")

def predict():
    """Generate predictions for test set"""
    device = Config.DEVICE
    test_loader = get_test_loader()
    
    # Load best model from each fold
    models = []
    for fold in range(Config.N_FOLDS):
        model = GrammarScoringModel().to(device)
        model.load_state_dict(
            torch.load(os.path.join(Config.OUTPUT_DIR, "models", f"best_model_fold_{fold + 1}.pt"))
        )
        model.eval()
        models.append(model)
    
    # Generate predictions
    all_preds = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            # Get predictions from all models
            fold_preds = []
            for model in models:
                pred = model(data)
                fold_preds.append(pred.cpu().numpy())
            # Average predictions
            ensemble_pred = np.mean(fold_preds, axis=0)
            all_preds.extend(ensemble_pred)
    
    # Create submission file
    test_df = pd.read_csv(Config.TEST_CSV)
    submission_df = pd.DataFrame({
        'filename': test_df['filename'],
        'label': all_preds
    })
    submission_df.to_csv(os.path.join(Config.OUTPUT_DIR, "submission.csv"), index=False)
    logger.info("Predictions saved to submission.csv")

if __name__ == "__main__":
    main()
    predict() 