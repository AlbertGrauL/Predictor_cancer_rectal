import torch
import torch.nn as nn
import mlflow
from tqdm import tqdm
from .utils import get_logger, save_checkpoint
from .metrics import calculate_metrics
from .config import DEVICE, MODELS_DIR

logger = get_logger("trainer")

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    for images, targets in tqdm(loader, desc="Training", leave=False):
        images, targets = images.to(device), targets.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        all_targets.append(targets.detach())
        all_outputs.append(torch.sigmoid(outputs).detach())
        
    avg_loss = running_loss / len(loader)
    metrics = calculate_metrics(torch.cat(all_targets), torch.cat(all_outputs))
    return avg_loss, metrics

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validation", leave=False):
            images, targets = images.to(device), targets.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            all_targets.append(targets.detach())
            all_outputs.append(torch.sigmoid(outputs).detach())
            
    avg_loss = running_loss / len(loader)
    metrics = calculate_metrics(torch.cat(all_targets), torch.cat(all_outputs))
    return avg_loss, metrics

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience, model_name):
    best_val_loss = float('inf')
    counter = 0
    history = []
    
    with mlflow.start_run(run_name=model_name, nested=True):
        mlflow.log_params({
            "epochs": epochs,
            "patience": patience,
            "batch_size": train_loader.batch_size,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "model_architecture": "efficientnet_b0"
        })
        
        for epoch in range(epochs):
            train_loss, train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_auc": train_metrics['auc'],
                "train_sensitivity": train_metrics['sensitivity'],
                "train_specificity": train_metrics['specificity'],
                "val_loss": val_loss,
                "val_auc": val_metrics['auc'],
                "val_sensitivity": val_metrics['sensitivity'],
                "val_specificity": val_metrics['specificity']
            }, step=epoch)
            
            log_msg = (f"Epoch {epoch+1}/{epochs} | "
                       f"Train Loss: {train_loss:.4f}, AUC: {train_metrics['auc']:.4f}, Sens: {train_metrics['sensitivity']:.4f} | "
                       f"Val Loss: {val_loss:.4f}, AUC: {val_metrics['auc']:.4f}, Sens: {val_metrics['sensitivity']:.4f}")
            logger.info(log_msg)
            
            # Early Stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                save_path = f"{MODELS_DIR}/{model_name}_best.pth"
                save_checkpoint(model, optimizer, epoch, save_path)
                mlflow.log_artifact(save_path, artifact_path="best_models")
                logger.info(f"--> Modelo guardado en {save_path}")
            else:
                counter += 1
                if counter >= patience:
                    logger.info("Early stopping activado.")
                    break
                    
            history.append({
                'epoch': epoch,
                'train_loss': train_loss, 'val_loss': val_loss,
                'train_auc': train_metrics['auc'], 'val_auc': val_metrics['auc']
            })
            
    return history
