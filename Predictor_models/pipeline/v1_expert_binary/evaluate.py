import torch
from torch.utils.data import DataLoader
from .dataset import EndoDataset
from .transforms import get_transforms
from .models import get_model
from .metrics import calculate_metrics, plot_confusion_matrix
from .utils import get_logger, load_checkpoint
from .config import DEVICE, MODELS_DIR, BATCH_SIZE, FIGURES_DIR

logger = get_logger("evaluate")

def evaluate_model(model_name, target_category, loader=None):
    logger.info(f"--- Evaluando Modelo: {model_name} para {target_category} ---")
    
    # 1. Preparar Datos de Test (si no se proporciona loader)
    if loader is None:
        test_transform = get_transforms(train=False)
        dataset = EndoDataset(target_category=target_category, transform=test_transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Cargar Modelo
    model = get_model("efficientnet_b0").to(DEVICE)
    checkpoint_path = f"{MODELS_DIR}/{model_name}_best.pth"
    
    try:
        load_checkpoint(model, None, checkpoint_path)
        logger.info(f"Cargado checkpoint desde {checkpoint_path}")
    except FileNotFoundError:
        logger.error(f"No se encontró el archivo {checkpoint_path}")
        return

    # 3. Inferencia
    model.eval()
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
            outputs = model(images)
            all_targets.append(targets.detach())
            all_outputs.append(torch.sigmoid(outputs).detach())
            
    # 4. Métricas Finales
    targets_cat = torch.cat(all_targets)
    outputs_cat = torch.cat(all_outputs)
    results = calculate_metrics(targets_cat, outputs_cat)
    
    # 5. Guardar Matriz de Confusión
    cm_path = FIGURES_DIR / f"confusion_matrix_{target_category}.png"
    plot_confusion_matrix(targets_cat, outputs_cat, target_category, cm_path)
    logger.info(f"Matriz de confusión guardada en: {cm_path}")
    
    logger.info(f"RESULTADOS FINALES ({target_category}):")
    logger.info(f"  AUC: {results['auc']:.4f}")
    logger.info(f"  Sensibilidad (Recall): {results['sensitivity']:.4f}")
    logger.info(f"  Especificidad: {results['specificity']:.4f}")
    logger.info(f"  Precisión: {results['precision']:.4f}")
    
    results['confusion_matrix_path'] = str(cm_path)
    return results

if __name__ == "__main__":
    evaluate_model("polipos", "polipos")
