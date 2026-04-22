from __future__ import annotations

import asyncio
import io
from pathlib import Path

from fastapi import UploadFile

from tests.conftest import import_fresh, make_module


def test_predict_endpoint_returns_probabilities_and_cleans_temp_file(monkeypatch, tmp_path):
    seen = {}

    class DummyPredictor:
        def predict(self, image_path):
            path = Path(image_path)
            seen["path"] = path
            seen["exists_during_predict"] = path.exists()
            return {"polipos": 0.91, "sangre": 0.02}

    make_module(monkeypatch, "Predictor_models.pipeline.inference", Predictor=DummyPredictor)
    module = import_fresh(monkeypatch, "Predictor_api.main")

    upload = UploadFile(
        filename="frame.jpg",
        file=io.BytesIO(b"fake-image-content"),
    )
    response = asyncio.run(module.predict(upload))

    assert response == {"polipos": 0.91, "sangre": 0.02}
    assert seen["exists_during_predict"] is True
    assert seen["path"].exists() is False


def test_app_has_expected_cors_origin(monkeypatch):
    class DummyPredictor:
        def predict(self, image_path):
            return {"negativos": 1.0}

    make_module(monkeypatch, "Predictor_models.pipeline.inference", Predictor=DummyPredictor)
    module = import_fresh(monkeypatch, "Predictor_api.main")

    cors_entries = [entry for entry in module.app.user_middleware if entry.cls.__name__ == "CORSMiddleware"]
    assert cors_entries
    assert cors_entries[0].kwargs["allow_origins"] == ["http://localhost:4200"]
