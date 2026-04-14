from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from .tabular_utils import transform_patient_payload
from ..utils import resolve_path


class TabularPredictor:
    def __init__(self, checkpoint_path: str | Path, metadata_path: str | Path) -> None:
        self.checkpoint_path = resolve_path(checkpoint_path)
        self.metadata_path = resolve_path(metadata_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        with self.checkpoint_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.model = payload["model"]
        self.class_names = payload["class_names"]
        self.preprocessing_signature = payload["preprocessing_signature"]

    def predict(self, patient_payload: dict[str, Any]) -> dict[str, Any]:
        features = transform_patient_payload(patient_payload, self.preprocessing_signature)
        probabilities = self.model.predict_proba(features)[0]
        predicted_index = int(probabilities.argmax())
        return {
            "predicted_class": self.class_names[predicted_index],
            "confidence": float(probabilities[predicted_index]),
            "probabilities": {
                class_name: float(probability) for class_name, probability in zip(self.class_names, probabilities.tolist())
            },
        }
