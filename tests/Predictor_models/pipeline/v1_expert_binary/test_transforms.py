from __future__ import annotations

import importlib

from PIL import Image
import torch


def test_train_transforms_return_normalized_tensor():
    module = importlib.import_module("Predictor_models.pipeline.v1_expert_binary.transforms")

    transform = module.get_transforms(train=True)
    output = transform(Image.new("RGB", (32, 32), color="white"))

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 224, 224)
    assert output.dtype == torch.float32


def test_eval_transforms_return_expected_shape():
    module = importlib.import_module("Predictor_models.pipeline.v1_expert_binary.transforms")

    transform = module.get_transforms(train=False)
    output = transform(Image.new("RGB", (48, 64), color="black"))

    assert output.shape == (3, 224, 224)
