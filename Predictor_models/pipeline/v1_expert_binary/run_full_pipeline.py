import gc
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import DataLoader, random_split
from Predictor_models.pipeline.v1_expert_binary.config import BATCH_SIZE, LEARNING_RATE, EPOCHS, EARLY_STOPPING_PATIENCE, DEVICE
from Predictor_models.pipeline.v1_expert_binary.dataset import EndoDataset
from Predictor_models.pipeline.v1_expert_binary.transforms import get_transforms
from Predictor_models.pipeline.v1_expert_binary.models import get_model
from Predictor_models.pipeline.v1_expert_binary.train import train_model
from Predictor_models.pipeline.v1_expert_binary.evaluate import evaluate_model
from Predictor_models.pipeline.v1_expert_binary.utils import get_logger

logger = get_logger("full_pipeline")

def train_single_category(target_category):
    logger.info(f"========== Entrenando Especialista para: {target_category.upper()} ==========")
    
    # Transformaciones
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    # Cargar dataset binario One-vs-Rest para esta categoría
    full_dataset = EndoDataset(target_category=target_category, transform=None)
    
    if len(full_dataset) == 0:
        logger.error(f"No se encontraron imágenes para {target_category}. Saltando...")
        return
        
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_data, val_data, test_data = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Asignamos temporalmente el transform a la clase subyacente.
    # En producción es mejor envolver el Subset, pero esto funciona para este pipeline estricto secuencial
    train_data.dataset.transform = train_transform
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    
    logger.info(f"Dataset {target_category}: {len(train_data)} entrenamiento, {len(val_data)} validación")
    
    # Inicializar arquitectura
    model = get_model("efficientnet_b0", pretrained=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    # El archivo train.py ya contiene el bloque mlflow.start_run()
    train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        criterion, 
        epochs=EPOCHS, 
        patience=EARLY_STOPPING_PATIENCE, 
        model_name=target_category
    )
    
    # --- EVALUACIÓN POST-ENTRENAMIENTO (Test Split) ---
    logger.info(f"Evaluando {target_category} en el conjunto de test...")
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # Abrimos el run de MLflow manualmente para loguear el artefacto de evaluación
    with mlflow.start_run(run_name=f"eval_{target_category}", nested=True):
        results = evaluate_model(target_category, target_category, loader=test_loader)
        if results and 'confusion_matrix_path' in results:
            mlflow.log_artifact(results['confusion_matrix_path'], artifact_path="evaluation_plots")
            
    logger.info(f"========== OK: Entrenamiento y Evaluación de {target_category} finalizados ==========")
    
    # Limpieza explícita
    del model, train_loader, val_loader, test_loader

def main():
    logger.info("INICIANDO ORQUESTADOR GLOBAL DE MODELOS CLINICOS")
    
    # Configurar el tracking de MLflow global
    mlflow.set_tracking_uri("sqlite:///mlflow.db") # Guarda los logs en una BBDD local estable
    mlflow.set_experiment("Predictor_Cancer_Endoscopico")
    
    categories = ["polipos", "sangre", "inflamacion", "negativos"]
    
    for cat in categories:
        try:
            train_single_category(cat)
        except Exception as e:
            logger.error(f"Fallo critico al entrenar {cat}: {str(e)}")
            
        # --- PREVENCIÓN DE OUT-OF-MEMORY (OOM) EN GPU ---
        # 1. Forzar la recolección de basura de Python
        gc.collect()
        # 2. Obligar a la GPU de NVIDIA a vaciar el caché remanente antes del siguiente modelo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"VRAM Limpiada tras procesar {cat}.")

if __name__ == "__main__":
    main()
