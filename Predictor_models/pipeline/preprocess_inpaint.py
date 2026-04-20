import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import ToTensor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "imagenes_cancer"
MASKS_DIR = BASE_DIR / "data" / "text_masks"

# Necesitamos importar AOT-GAN. Asumimos que esta al mismo nivel de tu proyecto
AOTGAN_REPO_DIR = Path(r"C:\Users\alber\AOT-GAN-for-Inpainting\src")
sys.path.append(str(AOTGAN_REPO_DIR))

try:
    from model.aotgan import InpaintGenerator
    class DummyArgs:
        rates = [1, 2, 4, 8]
        block_num = 8
except ImportError:
    print("Error: No se pudo importar AOT-GAN. Verifica que la ruta AOTGAN_REPO_DIR sea correcta.")
    sys.exit(1)

def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)

def process_inpainting(weights_path):
    print("Inicializando AOT-GAN en GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args = DummyArgs()
    model = InpaintGenerator(args).to(device)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Pesos preentrenados cargados con éxito.")
    else:
        print(f"Advertencia: No se encontraron pesos en {weights_path}.")
        print("Asegúrate de entrenar AOT-GAN en la Fase 2 antes de ejecutar esto de forma final.")
        return

    model.eval()
    
    # Buscar todas las máscaras generadas
    mask_files = list(MASKS_DIR.rglob("*.png"))
    print(f"Buscando limpiar {len(mask_files)} imágenes con texto...")
    
    for mpath in tqdm(mask_files):
        # Derivar ruta de imagen original
        rel_path = mpath.relative_to(MASKS_DIR)
        
        # Como las originales pueden ser jpg o png, probamos:
        orig_jpg = DATA_DIR / rel_path.with_suffix('.jpg')
        orig_png = DATA_DIR / rel_path.with_suffix('.png')
        
        orig_path = orig_jpg if orig_jpg.exists() else orig_png
        if not orig_path.exists():
            continue
            
        # Preparar destino
        out_rel_dir = orig_path.parent.relative_to(DATA_DIR)
        out_dir = DATA_DIR.parent / f"{out_rel_dir.parts[0]}_cleaned" / Path(*out_rel_dir.parts[1:])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / orig_path.name
        
        if out_path.exists():
            continue # Ya procesado
            
        # Inferencia
        image = ToTensor()(Image.open(orig_path).convert("RGB"))
        image = (image * 2.0 - 1.0).unsqueeze(0).to(device)
        
        mask = ToTensor()(Image.open(mpath).convert("L")).unsqueeze(0).to(device)
        
        # Redimensionar tensores temporalmente a múltiplos de 8 si es necesario (AOT-GAN tip)
        image_masked = image * (1 - mask) + mask

        with torch.no_grad():
            pred_img = model(image_masked, mask)

        comp_imgs = (1 - mask) * image + mask * pred_img
        
        postprocess(comp_imgs[0]).save(out_path)

if __name__ == "__main__":
    # La ruta al checkpoint que se generará tras el fine-tuning
    custom_weights = r"C:\Users\alber\AOT-GAN-for-Inpainting\experiments\colon_inpaint\latest_G.pth"
    process_inpainting(custom_weights)
