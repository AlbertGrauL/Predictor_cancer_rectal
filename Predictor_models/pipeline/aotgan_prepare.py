import os
import cv2
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "imagenes_cancer"
AOT_TRAIN_DIR = BASE_DIR / "data" / "aotgan_train"

CLEAN_IMG_DIR = AOT_TRAIN_DIR / "images" / "colon_clean"
MASKS_DIR = AOT_TRAIN_DIR / "masks" / "pconv"

def init_dirs():
    os.makedirs(CLEAN_IMG_DIR, exist_ok=True)
    os.makedirs(MASKS_DIR, exist_ok=True)

def gather_clean_images():
    """Recoge las imágenes set1 y set2 que ya sabemos que no tienen texto."""
    polipos_out_dir = DATA_DIR / "Polipos" / "imagenes con polipos destacados" / "output" / "original"
    print("Recopilando imágenes limpias conocidas de set1 y set2...")
    count = 0
    if polipos_out_dir.exists():
        for filename in os.listdir(polipos_out_dir):
            if filename.startswith("set1") or filename.startswith("set2"):
                src = polipos_out_dir / filename
                dst = CLEAN_IMG_DIR / filename
                if not dst.exists():
                    shutil.copy2(src, dst)
                count += 1
    print(f"✅ Encontradas/copiadas {count} imágenes completas sin texto.")
    return count

def generate_cropped_clean_images(target_total=3000, current_count=0):
    """Extrae la mitad derecha de las imágenes crudas para asegurar que no entra el texto."""
    unclassified_dir = DATA_DIR / "imagenes sin clasificar" / "images"
    remaining = target_total - current_count
    
    if remaining <= 0 or not unclassified_dir.exists():
        return
        
    print(f"Generando {remaining} recortes de imágenes (ignorando el texto del lado izquierdo)...")
    images_list = list(unclassified_dir.glob("*.jpg"))
    np.random.seed(42) # Replicabilidad
    np.random.shuffle(images_list)
    
    added = 0
    for img_path in tqdm(images_list, desc="Cortando imágenes fuertes", leave=False):
        if added >= remaining:
            break
        
        try:
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            h, w = img.shape[:2]
            # La marca de agua está a la izquierda. Tomamos la mitad derecha (del 40% al 100%)
            left_margin = int(w * 0.4)
            safe_region = img[:, left_margin:]
            
            sh_h, sh_w = safe_region.shape[:2]
            if sh_w < 256 or sh_h < 256:
                continue
                
            # Extraer un parche de la zona central de la safe_region para max calidad y enfocar la mucosa
            sz = min(sh_w, sh_h, 512)
            c_x, c_y = sh_w // 2, sh_h // 2
            x1, y1 = c_x - sz // 2, c_y - sz // 2
            patch = safe_region[y1:y1+sz, x1:x1+sz]
            
            # AOT-GAN usa 512x512
            patch = cv2.resize(patch, (512, 512))
            
            dst_name = f"crop_clean_{added:04d}.jpg"
            cv2.imwrite(str(CLEAN_IMG_DIR / dst_name), patch)
            added += 1
            
        except Exception as e:
            continue
            
    print(f"✅ Generados {added} recortes adicionales de colon.")

def generate_random_masks(num_masks=1000):
    """Genera las máscaras aleatorias gruesas e irregulares ("pconv" strokes) necesarias para entrenar AOT-GAN."""
    print(f"Generando {num_masks} máscaras aleatorias (random strokes) para entrenamiento...")
    for i in tqdm(range(num_masks), desc="Máscaras Sintéticas", leave=False):
        mask = np.zeros((512, 512), np.uint8)
        
        # Rayones
        num_strokes = np.random.randint(3, 10)
        for _ in range(num_strokes):
            x1, y1 = np.random.randint(0, 512), np.random.randint(0, 512)
            x2, y2 = np.random.randint(0, 512), np.random.randint(0, 512)
            thickness = np.random.randint(15, 60)
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)
            
        # Círculos / Bloques grandes
        num_circles = np.random.randint(1, 5)
        for _ in range(num_circles):
            cx, cy = np.random.randint(0, 512), np.random.randint(0, 512)
            radius = np.random.randint(30, 80)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            
        cv2.imwrite(str(MASKS_DIR / f"mask_{i:04d}.png"), mask)

if __name__ == "__main__":
    init_dirs()
    count = gather_clean_images()
    generate_cropped_clean_images(target_total=3000, current_count=count)
    generate_random_masks(num_masks=1000)
    print("¡Fase 1 completada con éxito! El dataset está listo para entrenar AOT-GAN.")
