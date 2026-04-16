import logging
import os
import torch
from datetime import datetime
from .config import LOGS_DIR

def get_logger(name):
    """Configura y devuelve un logger con salida a consola y archivo."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Formato de logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Handler para Consola
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Handler para Archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(os.path.join(LOGS_DIR, f"train_{timestamp}.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def save_checkpoint(model, optimizer, epoch, path):
    """Guarda un checkpoint del modelo."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    """Carga un checkpoint del modelo."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
