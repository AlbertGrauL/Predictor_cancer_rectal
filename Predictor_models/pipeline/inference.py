from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .gradcam import generate_gradcam
from .models import build_model
from .transforms import build_transforms


class Predictor:
    def __init__(self, checkpoint_path: str | Path, metadata_path: str | Path, image_size: int = 224) -> None:
        try:
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Faltan dependencias de inferencia. Ejecuta `uv sync`.") from exc

        self.torch = torch
        self.Image = Image
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = Path(metadata_path)
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.class_names = metadata["class_names"]
        self.model_name = metadata["model_name"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.eval_transform = build_transforms(image_size)

        self.model = build_model(self.model_name, len(self.class_names), pretrained=False)
        payload = torch.load(self.checkpoint_path, map_location=self.device)
        state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        image = self.Image.open(image_path).convert("RGB")
        tensor = self.eval_transform(image).unsqueeze(0).to(self.device)
        with self.torch.enable_grad():
            result = generate_gradcam(self.model, tensor)
        predicted_class = self.class_names[result.predicted_index]
        return {
            "predicted_class": predicted_class,
            "confidence": result.probabilities[result.predicted_index],
            "probabilities": {
                class_name: probability for class_name, probability in zip(self.class_names, result.probabilities)
            },
            "heatmap": result.heatmap,
        }
