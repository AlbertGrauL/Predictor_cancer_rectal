import torch
from .models import get_model
from .transforms import get_transforms
from .config import MODELS_DIR, DEVICE, INPAINT_WEIGHTS
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor
from .lib.aotgan import InpaintGenerator
from .preprocess_masks import generate_text_mask

class InpaintArgs:
    rates = [1, 2, 4, 8]
    block_num = 8

class Predictor:
    """Consolida los 4 modelos para realizar un diagnóstico completo."""
    def __init__(self):
        self.categories = ["polipos", "sangre", "inflamacion", "negativos"]
        self.models = {}
        self.transform = get_transforms(train=False)
        
        for cat in self.categories:
            model_path = MODELS_DIR / f"{cat}_best.pth"
            if model_path.exists():
                model = get_model("efficientnet_b0").to(DEVICE)
                checkpoint = torch.load(model_path, map_location=DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                self.models[cat] = model
                print(f"Modelo cargado: {cat}")
            else:
                print(f"Advertencia: Modelo para {cat} no encontrado en {model_path}")
        
        # ── Inpainting Model Initialization ─────────────────────────────────
        if INPAINT_WEIGHTS.exists():
            self.inpaint_model = InpaintGenerator(InpaintArgs()).to(DEVICE)
            self.inpaint_model.load_state_dict(torch.load(INPAINT_WEIGHTS, map_location=DEVICE))
            self.inpaint_model.eval()
            print("Modelo Inpainting cargado correctamente.")
        else:
            self.inpaint_model = None
            print("Advertencia: No se encontró el modelo de Inpainting.")

    def predict(self, image_path):
        # 1. Cargar Imagen original
        img_raw = Image.open(image_path).convert("RGB")
        w_orig, h_orig = img_raw.size
        
        # 2. Preprocesado / Inpainting (si hay texto detectado)
        img_cleaned = self._clean_image(image_path, img_raw)
        
        # 3. Transformar para clasificación (EfficientNet)
        input_tensor = self.transform(img_cleaned).unsqueeze(0).to(DEVICE)
        
        results = {}
        with torch.no_grad():
            for cat, model in self.models.items():
                logit = model(input_tensor)
                prob = torch.sigmoid(logit).item()
                results[cat] = prob
                
        return results

    def _clean_image(self, image_path, pil_img):
        """Aplica inpainting si se detecta texto clínico."""
        if self.inpaint_model is None:
            return pil_img
            
        # Generar máscara de texto (temporal)
        mask_path = "temp_inference_mask.png"
        if not generate_text_mask(image_path, mask_path):
            return pil_img # No hay texto, devolver original
            
        # Inpaint Inferencia
        w_orig, h_orig = pil_img.size
        img_512 = pil_img.resize((512, 512))
        mask_512 = Image.open(mask_path).convert("L").resize((512, 512))
        
        img_t = (ToTensor()(img_512) * 2.0 - 1.0).unsqueeze(0).to(DEVICE)
        mask_t = ToTensor()(mask_512).unsqueeze(0).to(DEVICE)
        
        img_masked = img_t * (1 - mask_t) + mask_t
        with torch.no_grad():
            pred_t = self.inpaint_model(img_masked, mask_t)
            
        comp_t = (1 - mask_t) * img_t + mask_t * pred_t
        comp_t = torch.clamp(comp_t, -1.0, 1.0)
        comp_img = ((comp_t[0].permute(1, 2, 0).cpu().numpy() + 1) / 2.0 * 255.0).astype(np.uint8)
        
        return Image.fromarray(comp_img).resize((w_orig, h_orig))

if __name__ == "__main__":
    # Ejemplo de uso
    # predictor = Predictor()
    # results = predictor.predict("ruta/a/imagen.jpg")
    # print(results)
    pass
