import os
from pathlib import Path

# --- Rutas del Proyecto ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "Predictor_models" / "data" / "imagenes_cancer"

# Categorías y Carpetas Específicas
PATHS = {
    "polipos": DATA_DIR / "Polipos" / "imagenes con polipos destacados" / "original",
    "sangre": DATA_DIR / "Sangre_Paredes" / "sangre_activa",
    "inflamacion": DATA_DIR / "Sangre_Paredes" / "inflamacion_leve",
    "negativos": DATA_DIR / "Casos_negativos"
}

# --- Hiperparámetros de Entrenamiento ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7

# --- Split de Datos ---
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# --- Configuración de Dispositivo ---
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Logging y Artefactos ---
LOGS_DIR = BASE_DIR / "Predictor_models" / "logs"
MODELS_DIR = BASE_DIR / "Predictor_models" / "artifacts" / "models"
FIGURES_DIR = BASE_DIR / "Predictor_models" / "artifacts" / "figures"
EXPERIMENTS_DIR = BASE_DIR / "Predictor_models" / "experiments"
INPAINT_WEIGHTS = MODELS_DIR / "G0100000.pt"

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
