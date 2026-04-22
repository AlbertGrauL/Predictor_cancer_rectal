from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from tests.conftest import import_fresh, make_module


class _FakeProbModel:
    def predict_proba(self, frame):
        return np.array([[0.25, 0.75]])


def test_tabular_predictor_loads_artifacts_and_predicts(monkeypatch, tmp_path):
    make_module(monkeypatch, "Predictor_models.pipeline.v2_multiclass.utils", resolve_path=lambda value: Path(value))
    module = import_fresh(monkeypatch, "Predictor_models.pipeline.v2_multiclass.tabular.tabular_inference")

    checkpoint_path = tmp_path / "model.pkl"
    metadata_path = tmp_path / "metadata.json"
    checkpoint_path.write_bytes(
        pickle.dumps(
            {
                "model": _FakeProbModel(),
                "class_names": ["sin_riesgo_clinico", "riesgo_clinico"],
                "preprocessing_signature": {
                    "encoded_feature_columns": [
                        "age",
                        "sex",
                        "sof",
                        "alcohol",
                        "digestive_family_history_no",
                        "digestive_family_history_colon",
                        "digestive_family_history_gastric",
                        "digestive_family_history_other_positive",
                        "digestive_family_history_unknown_dirty",
                    ],
                    "binary_columns": ["sof"],
                    "ordinal_columns": ["alcohol"],
                    "family_history_categories": ["no", "colon", "gastric", "other_positive", "unknown_dirty"],
                },
            }
        )
    )
    metadata_path.write_text(json.dumps({"modality": "tabular"}), encoding="utf-8")

    predictor = module.TabularPredictor(checkpoint_path, metadata_path)
    result = predictor.predict(
        {"age": 64, "sex": 1, "sof": 1, "alcohol": 2, "digestive_family_history": "colon"}
    )

    assert predictor.metadata["modality"] == "tabular"
    assert result["predicted_class"] == "riesgo_clinico"
    assert result["confidence"] == 0.75
    assert result["probabilities"]["sin_riesgo_clinico"] == 0.25
