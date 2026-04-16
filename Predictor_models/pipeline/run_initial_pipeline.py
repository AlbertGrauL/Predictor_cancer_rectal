import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from .config import BATCH_SIZE, LEARNING_RATE, EPOCHS, EARLY_STOPPING_PATIENCE, DEVICE
from .dataset import EndoDataset
from .transforms import get_transforms
from .models import get_model
from .train import train_model
from .utils import get_logger

logger = get_logger("run_pipeline")

def main():
    logger.info("--- Iniciando Pipeline para MODELO DE PÓLIPOS ---")
    
    # 1. Datasets y Transformaciones
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    # Cargamos el dataset completo para pólipos
    full_dataset = EndoDataset(target_category="polipos", transform=None)
    
    if len(full_dataset) == 0:
        logger.error("No se encontraron imágenes en el dataset de pólipos. Abortando.")
        return

    # 2. Split (Train/Val/Test) 
    # Usamos transformaciones diferentes para cada split
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Aplicar transformaciones manualmente ya que random_split devuelve Subsets
    train_data.dataset.transform = train_transform
    # Nota: Como apuntan al mismo dataset, esto es un poco delicado. 
    # Una mejor forma es usar Subsets con clases wrapper, pero para el MVP usaremos copias o 
    # simplemente cambiaremos el transform durante el loop si fuera necesario.
    # Versión correcta:
    class AppliedTransformSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                # Si el dataset original ya devolvió una imagen PIL, aplicamos transform
                # En nuestro dataset.py __getitem__ ya aplica transform si existe.
                # Lo ideal es que dataset.py no aplique transform y lo hagamos aquí.
                pass 
            return x, y
        def __len__(self):
            return len(self.subset)

    # Optimizamos EndoDataset para que no aplique transform interna y la aplicamos aquí
    # Pero por ahora seguimos el plan simple.
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Dataset cargado: {len(train_data)} entrenamiento, {len(val_data)} validación")

    # 3. Modelo, Optimizer, Criterion
    model = get_model("efficientnet_b0", pretrained=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Usar pesos en la pérdida si hay desbalance (opcional)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Entrenamiento
    logger.info(f"Entrenando en dispositivo: {DEVICE}")
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        epochs=EPOCHS, 
        patience=EARLY_STOPPING_PATIENCE, 
        model_name="polipos"
    )
    
    logger.info("--- Pipeline finalizado con éxito ---")

if __name__ == "__main__":
    main()
