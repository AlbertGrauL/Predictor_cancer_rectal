from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from tests.conftest import import_fresh, make_module


def test_predictor_loads_existing_models_and_predicts(monkeypatch, tmp_path):
    loaded_states = []

    class FakeModel:
        def __init__(self):
            self.loaded = None

        def to(self, device):
            return self

        def load_state_dict(self, state):
            self.loaded = state
            loaded_states.append(state)

        def eval(self):
            return self

        def __call__(self, tensor):
            return torch.tensor([0.0])

    def fake_get_model(_name):
        return FakeModel()

    def fake_get_transforms(train=False):
        return lambda image: torch.ones(3, 8, 8)

    checkpoints_dir = tmp_path / "models"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "polipos_best.pth").write_bytes(b"x")
    (checkpoints_dir / "negativos_best.pth").write_bytes(b"y")

    make_module(monkeypatch, "Predictor_models.pipeline.v1_expert_binary.models", get_model=fake_get_model)
    make_module(monkeypatch, "Predictor_models.pipeline.v1_expert_binary.transforms", get_transforms=fake_get_transforms)
    make_module(
        monkeypatch,
        "Predictor_models.pipeline.v1_expert_binary.config",
        MODELS_DIR=checkpoints_dir,
        DEVICE="cpu",
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"model_state_dict": {"weights": 1}})

    module = import_fresh(monkeypatch, "Predictor_models.pipeline.v1_expert_binary.inference")
    predictor = module.Predictor()

    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (16, 16), color="blue").save(image_path)
    results = predictor.predict(image_path)

    assert set(predictor.models) == {"polipos", "negativos"}
    assert len(loaded_states) == 2
    assert set(results) == {"polipos", "negativos"}
    assert all(0.0 <= value <= 1.0 for value in results.values())
