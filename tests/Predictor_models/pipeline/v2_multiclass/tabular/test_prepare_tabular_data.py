from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from tests.conftest import import_fresh, make_module


def test_parse_args_uses_default_config(monkeypatch):
    make_module(monkeypatch, "Predictor_models.pipeline.v2_multiclass.config", load_config=lambda value: value)
    make_module(monkeypatch, "Predictor_models.pipeline.v2_multiclass.utils", load_paths=lambda cfg: cfg, write_csv=lambda *args: None, write_json=lambda *args: None)
    make_module(
        monkeypatch,
        "Predictor_models.pipeline.v2_multiclass.tabular.tabular_utils",
        prepare_tabular_dataframe=lambda cfg: None,
        split_dataframe=lambda frame, cfg: frame,
    )
    monkeypatch.setattr("sys.argv", ["prepare_tabular_data"])
    module = import_fresh(monkeypatch, "Predictor_models.pipeline.v2_multiclass.tabular.prepare_tabular_data")

    args = module.parse_args()
    assert args.config.endswith("tabular_baseline.yaml")


def test_main_writes_manifest_and_split_summary(monkeypatch, tmp_path):
    written = {}

    def fake_write_csv(path, rows, columns):
        written["csv_path"] = path
        written["rows"] = rows
        written["columns"] = columns

    def fake_write_json(path, payload):
        written["json_path"] = path
        written["json_payload"] = payload

    config = {
        "dataset": {
            "id_column": "patient_id",
            "binary_columns": ["sof"],
            "ordinal_columns": ["alcohol"],
            "target_column": "target",
        }
    }
    paths = SimpleNamespace(manifests_dir=tmp_path, reports_dir=tmp_path)
    artifacts = SimpleNamespace(
        dataframe=pd.DataFrame(),
        feature_columns=[
            "age",
            "sex",
            "sof",
            "alcohol",
            "digestive_family_history_colon",
            "digestive_family_history_no",
        ],
    )
    prepared = pd.DataFrame(
        [
            {
                "row_index": 0,
                "patient_id": 100,
                "age": 52,
                "sex": 1,
                "sof": 1,
                "alcohol": 2,
                "digestive_family_history": "colon",
                "digestive_family_history_group": "colon",
                "digestive_family_history_colon": 1,
                "digestive_family_history_no": 0,
                "target": 1,
                "split": "train",
            },
            {
                "row_index": 1,
                "patient_id": 101,
                "age": 48,
                "sex": 0,
                "sof": 0,
                "alcohol": 0,
                "digestive_family_history": "no",
                "digestive_family_history_group": "no",
                "digestive_family_history_colon": 0,
                "digestive_family_history_no": 1,
                "target": 0,
                "split": "test",
            },
        ]
    )

    make_module(monkeypatch, "Predictor_models.pipeline.v2_multiclass.config", load_config=lambda _: config)
    make_module(
        monkeypatch,
        "Predictor_models.pipeline.v2_multiclass.utils",
        load_paths=lambda _: paths,
        write_csv=fake_write_csv,
        write_json=fake_write_json,
    )
    make_module(
        monkeypatch,
        "Predictor_models.pipeline.v2_multiclass.tabular.tabular_utils",
        prepare_tabular_dataframe=lambda _: artifacts,
        split_dataframe=lambda df, cfg: prepared,
    )
    module = import_fresh(monkeypatch, "Predictor_models.pipeline.v2_multiclass.tabular.prepare_tabular_data")
    monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(config="cfg.yaml"))

    module.main()

    assert written["csv_path"].name == "tabular_manifest.csv"
    assert written["json_path"].name == "tabular_split_summary.json"
    assert written["json_payload"]["train"]["rows"] == 1
    assert written["rows"][0]["patient_id"] == 100
