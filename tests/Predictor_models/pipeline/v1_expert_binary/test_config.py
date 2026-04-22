from __future__ import annotations

import importlib


def test_config_exposes_expected_paths_and_constants():
    module = importlib.import_module("Predictor_models.pipeline.v1_expert_binary.config")

    assert module.BASE_DIR.exists()
    assert module.DATA_DIR.name == "imagenes_cancer"
    assert set(module.PATHS) == {"polipos", "sangre", "inflamacion", "negativos"}
    assert module.IMG_SIZE == (224, 224)
    assert module.TRAIN_SPLIT + module.VAL_SPLIT + module.TEST_SPLIT == 1.0
    assert module.DEVICE in {"cpu", "cuda"}
    assert module.LOGS_DIR.exists()
    assert module.MODELS_DIR.exists()
    assert module.EXPERIMENTS_DIR.exists()
