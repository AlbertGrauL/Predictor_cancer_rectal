from __future__ import annotations

import json
from types import SimpleNamespace

from tests.conftest import import_fresh, make_module


def test_main_exports_only_tabular_experiments(monkeypatch, tmp_path):
    metrics_dir = tmp_path / "metrics"
    reports_dir = tmp_path / "reports"
    metrics_dir.mkdir()
    reports_dir.mkdir()

    (metrics_dir / "xgboost_evaluation.json").write_text(
        json.dumps(
            {
                "model_name": "xgboost",
                "modality": "tabular",
                "metrics": {
                    "accuracy": 0.9,
                    "precision_positive": 0.8,
                    "recall_positive": 0.7,
                    "f1_positive": 0.75,
                    "roc_auc": 0.95,
                    "pr_auc": 0.91,
                },
                "training_setup": {"cv_best_score": 0.81, "calibration": {"method": "sigmoid"}},
            }
        ),
        encoding="utf-8",
    )
    (metrics_dir / "resnet_evaluation.json").write_text(
        json.dumps({"model_name": "resnet", "modality": "image", "metrics": {"accuracy": 0.8}}),
        encoding="utf-8",
    )

    make_module(monkeypatch, "Predictor_models.pipeline.v2_multiclass.config", load_config=lambda _: {})
    make_module(
        monkeypatch,
        "Predictor_models.pipeline.v2_multiclass.utils",
        load_paths=lambda _: SimpleNamespace(metrics_dir=metrics_dir, reports_dir=reports_dir),
    )
    module = import_fresh(monkeypatch, "Predictor_models.pipeline.v2_multiclass.tabular.summarize_tabular_experiments")
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(config="cfg.yaml"))

    module.main()

    output = (reports_dir / "experiment_summary_tabular.csv").read_text(encoding="utf-8")
    assert "xgboost" in output
    assert "resnet" not in output
