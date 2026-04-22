from __future__ import annotations

import importlib
from pathlib import Path

import torch


def test_get_logger_creates_console_and_file_handlers(monkeypatch, tmp_path):
    module = importlib.import_module("Predictor_models.pipeline.v1_expert_binary.utils")
    monkeypatch.setattr(module, "LOGS_DIR", tmp_path)

    logger = module.get_logger("unit-test-logger")

    assert len(logger.handlers) == 2
    file_handlers = [handler for handler in logger.handlers if hasattr(handler, "baseFilename")]
    assert file_handlers
    assert Path(file_handlers[0].baseFilename).parent == tmp_path


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    module = importlib.import_module("Predictor_models.pipeline.v1_expert_binary.utils")
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint_path = tmp_path / "checkpoint.pt"

    initial_weight = model.weight.detach().clone()
    module.save_checkpoint(model, optimizer, epoch=7, path=checkpoint_path)

    with torch.no_grad():
        model.weight.fill_(0)

    epoch = module.load_checkpoint(model, optimizer, checkpoint_path)

    assert epoch == 7
    assert torch.allclose(model.weight.detach(), initial_weight)
