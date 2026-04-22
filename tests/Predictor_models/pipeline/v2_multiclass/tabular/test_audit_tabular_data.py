from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import import_fresh, make_module


def test_main_writes_audit_and_question_specs(monkeypatch, tmp_path):
    calls = []
    config = {"dataset": {"target_column": "target"}}
    paths = SimpleNamespace(reports_dir=tmp_path)
    artifacts = SimpleNamespace(questions="QUESTIONS")

    def fake_write_json(path, payload):
        calls.append((path.name, payload))

    make_module(monkeypatch, "Predictor_models.pipeline.v2_multiclass.config", load_config=lambda _: config)
    make_module(
        monkeypatch,
        "Predictor_models.pipeline.v2_multiclass.utils",
        load_paths=lambda _: paths,
        write_json=fake_write_json,
    )
    make_module(
        monkeypatch,
        "Predictor_models.pipeline.v2_multiclass.tabular.tabular_utils",
        build_question_specs=lambda cfg, questions: [{"variable": "age"}],
        build_tabular_audit_report=lambda cfg, art: {"rows": 10},
        prepare_tabular_dataframe=lambda cfg: artifacts,
    )

    module = import_fresh(monkeypatch, "Predictor_models.pipeline.v2_multiclass.tabular.audit_tabular_data")
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(config="cfg.yaml"))

    module.main()

    assert calls[0][0] == "tabular_dataset_audit.json"
    assert calls[0][1]["rows"] == 10
    assert calls[1][0] == "tabular_question_specs.json"
    assert calls[1][1]["questions"][0]["variable"] == "age"
