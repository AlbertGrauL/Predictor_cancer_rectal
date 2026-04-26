import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Ruta al repo de AOT-GAN
AOTGAN_REPO_DIR = Path(r"C:\Users\alber\AOT-GAN-for-Inpainting\src")
sys.path.append(str(AOTGAN_REPO_DIR))

from model.aotgan import InpaintGenerator
from preprocess_masks import generate_text_mask
from torchvision.transforms import ToTensor

class DummyArgs:
    rates = [1, 2, 4, 8]
    block_num = 8

def test_single_image(img_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # 1. Cargar Modelo
    model = InpaintGenerator(DummyArgs()).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 2. Generar Máscara temporalmente
    mask_path = "temp_mask.png"
    if not generate_text_mask(img_path, mask_path):
        print("No se detectó texto en esta imagen.")
        return
    
    # 3. Preparar entrada
    image_raw = Image.open(img_path).convert("RGB")
    w_orig, h_orig = image_raw.size
    
    # Redimensionar a 512x512 para el modelo si es necesario
    image_pil = image_raw.resize((512, 512))
    mask_pil = Image.open(mask_path).convert("L").resize((512, 512))
    
    image_t = (ToTensor()(image_pil) * 2.0 - 1.0).unsqueeze(0).to(device)
    mask_t = ToTensor()(mask_pil).unsqueeze(0).to(device)
    
    # 4. Inferencia
    image_masked = image_t * (1 - mask_t) + mask_t
    with torch.no_grad():
        pred_t = model(image_masked, mask_t)
    
    # 5. Composición y Postproceso
    comp_t = (1 - mask_t) * image_t + mask_t * pred_t
    comp_t = torch.clamp(comp_t, -1.0, 1.0)
    comp_img = ((comp_t[0].permute(1, 2, 0).cpu().numpy() + 1) / 2.0 * 255.0).astype(np.uint8)
    
    # Volver al tamaño original
    final_img = Image.fromarray(comp_img).resize((w_orig, h_orig))
    
    # Guardar comparativa
    res = Image.new('RGB', (w_orig * 2, h_orig))
    res.paste(image_raw, (0, 0))
    res.paste(final_img, (w_orig, 0))
    
    output_path = "test_inpainting_result.png"
    res.save(output_path)
    print(f"Resultado visual guardado en: {output_path}")

if __name__ == "__main__":
    # Prueba con una imagen del dataset
    img_path = r"C:\Users\alber\Desktop\Predictor_cancer_rectal\Predictor_models\data\imagenes_cancer\Polipos\imagenes con polipos destacados\original\set3_0051.jpg"
    weights = r"C:\Users\alber\Desktop\Predictor_cancer_rectal\Predictor_models\artifacts\models\G0100000.pt"
    
    if Path(img_path).exists():
        test_single_image(img_path, weights)
    else:
        print(f"No se encontró la imagen de prueba: {img_path}")
