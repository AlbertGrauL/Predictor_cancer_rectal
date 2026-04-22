from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests.conftest import import_fresh, make_module


def test_parse_args_supports_flags(monkeypatch):
    make_module(monkeypatch, "Predictor_models.pipeline.v2_multiclass.config", load_config=lambda value: value)
    make_module(monkeypatch, "Predictor_models.pipeline.v2_multiclass.utils", resolve_path=lambda value: Path("."))
    monkeypatch.setattr("sys.argv", ["run_tabular_model_comparison", "--skip-audit", "--models", "xgboost"])
    module = import_fresh(monkeypatch, "Predictor_models.pipeline.v2_multiclass.tabular.run_tabular_model_comparison")

    args = module.parse_args()
    assert args.skip_audit is True
    assert args.models == ["xgboost"]


def test_main_runs_expected_steps(monkeypatch, tmp_path):
    executed = []
    config = {"models": {"candidates": ["random_forest", "xgboost"]}}

    make_module(monkeypatch, "Predictor_models.pipeline.v2_multiclass.config", load_config=lambda _: config)
    make_module(monkeypatch, "Predictor_models.pipeline.v2_multiclass.utils", resolve_path=lambda _: tmp_path)
    module = import_fresh(monkeypatch, "Predictor_models.pipeline.v2_multiclass.tabular.run_tabular_model_comparison")

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(
            config="cfg.yaml",
            no_clean=False,
            skip_audit=False,
            skip_prepare=False,
            models=["xgboost"],
        ),
    )
    monkeypatch.setattr(module, "run_step", lambda command, title, cwd: executed.append((title, command, cwd)))

    module.main()

    titles = [title for title, _command, _cwd in executed]
    assert titles == [
        "0. Limpieza tabular",
        "1. Auditoría tabular",
        "2. Preparación tabular",
        "3.1 Entrenamiento de xgboost",
        "4.1 Evaluación de xgboost",
        "5. Resumen tabular",
    ]
    assert all(cwd == tmp_path for _title, _command, cwd in executed)
