import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "Predictor_models" / "data" / "imagenes_cancer"
MASKS_OUT_DIR = DATA_DIR.parent / "text_masks"

def init_dirs():
    os.makedirs(MASKS_OUT_DIR, exist_ok=True)

def generate_text_mask(img_path, save_path):
    """
    Genera una máscara del texto (blanco/verde brillante) que suele
    estar en el lado izquierdo de la imagen endoscópica.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False
        
    h, w = img.shape[:2]
    
    # Trabajamos solo en la mitad izquierda para evitar atrapar brillos
    # especulares de la mucosa en el centro/derecha.
    roi_w = int(w * 0.4)
    roi = img[:, :roi_w]
    
    # 1. Máscara para blancos brillantes (luminosidad alta)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask_white = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    
    # 2. Máscara para verdes brillantes (espacio HSV)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Verde en HSV (tonos entre 40 y 80, alta saturación y valor)
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Combinar ambas máscaras
    combined_mask = cv2.bitwise_or(mask_white, mask_green)
    
    # Dilatar la máscara para cubrir bordes antialias
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    
    # Crear una máscara final del tamaño de la imagen original
    final_mask = np.zeros((h, w), dtype=np.uint8)
    final_mask[:, :roi_w] = dilated_mask
    
    # Criterio: solo guardar si hay algo de texto significativo
    if np.sum(final_mask > 0) > 50:
        cv2.imwrite(str(save_path), final_mask)
        return True
    return False

def process_all_images():
    print("Escaneando dataset en busca de marcas de agua/texto...")
    # Buscamos en todas las carpetas, excepto 'aotgan_train' o '_cleaned'
    all_images = glob(str(DATA_DIR / "**/*.jpg"), recursive=True) + glob(str(DATA_DIR / "**/*.png"), recursive=True)
    
    valid_count = 0
    for img_path in tqdm(all_images):
        if "aotgan_train" in img_path or "text_masks" in img_path or "_cleaned" in img_path or "Polipos/imagenes con polipos destacados/original" in img_path:
            continue # Evitar procesar lo ya limpio
            
        rel_path = Path(img_path).relative_to(DATA_DIR)
        save_path = MASKS_OUT_DIR / rel_path.with_suffix('.png')
        
        # Crear subdirectorios si no existen
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if generate_text_mask(img_path, save_path):
            valid_count += 1
            
    print(f"✅ Se han generado {valid_count} máscaras de texto.")

if __name__ == "__main__":
    init_dirs()
    process_all_images()
