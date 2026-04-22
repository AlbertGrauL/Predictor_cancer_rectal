from __future__ import annotations

import importlib

import torch
import pytest


def test_calculate_metrics_for_mixed_predictions():
    module = importlib.import_module("Predictor_models.pipeline.v1_expert_binary.metrics")

    metrics = module.calculate_metrics(
        torch.tensor([1, 0, 1, 0]),
        torch.tensor([0.9, 0.2, 0.4, 0.8]),
        threshold=0.5,
    )

    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["precision"] == 0.5
    assert 0.0 <= metrics["auc"] <= 1.0


def test_calculate_metrics_falls_back_when_auc_is_undefined(monkeypatch):
    module = importlib.import_module("Predictor_models.pipeline.v1_expert_binary.metrics")
    monkeypatch.setattr(module, "roc_auc_score", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad auc")))

    metrics = module.calculate_metrics(
        torch.tensor([1, 1, 1]),
        torch.tensor([0.7, 0.8, 0.9]),
    )

    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 0
    assert metrics["auc"] == 0.5
