import torch
from .models import get_model
from .transforms import get_transforms
from .config import MODELS_DIR, DEVICE
from PIL import Image

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

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        
        results = {}
        with torch.no_grad():
            for cat, model in self.models.items():
                logit = model(input_tensor)
                prob = torch.sigmoid(logit).item()
                results[cat] = prob
                
        return results

if __name__ == "__main__":
    # Ejemplo de uso
    # predictor = Predictor()
    # results = predictor.predict("ruta/a/imagen.jpg")
    # print(results)
    pass
