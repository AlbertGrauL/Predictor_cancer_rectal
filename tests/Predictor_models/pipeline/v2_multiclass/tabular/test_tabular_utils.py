from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import import_fresh, make_module


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str], delimiter: str = ";") -> None:
    with path.open("w", encoding="latin1", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def tabular_module(monkeypatch):
    make_module(monkeypatch, "Predictor_models.pipeline.v2_multiclass.utils", resolve_path=lambda value: Path(value))
    return import_fresh(monkeypatch, "Predictor_models.pipeline.v2_multiclass.tabular.tabular_utils")


@pytest.fixture
def sample_config(tmp_path):
    data_csv = tmp_path / "data.csv"
    questions_csv = tmp_path / "questions.csv"

    feature_columns = [
        "age",
        "sex",
        "sof",
        "diabetes",
        "tenesmus",
        "previous_rt",
        "rectorrhagia",
        "alcohol",
        "tobacco",
        "intestinal_habit",
        "digestive_family_history",
    ]
    rows = [
        {
            "patient_id": 1,
            "target": "no",
            "age": 45,
            "sex": "woman",
            "sof": "yes",
            "diabetes": "no",
            "tenesmus": "yes",
            "previous_rt": "no",
            "rectorrhagia": "yes",
            "alcohol": 2,
            "tobacco": 1,
            "intestinal_habit": 3,
            "digestive_family_history": "colon cancer",
        },
        {
            "patient_id": 2,
            "target": "yes",
            "age": 67,
            "sex": "man",
            "sof": "no",
            "diabetes": "yes",
            "tenesmus": "no",
            "previous_rt": "yes",
            "rectorrhagia": "no",
            "alcohol": 1,
            "tobacco": 0,
            "intestinal_habit": 2,
            "digestive_family_history": "no",
        },
        {
            "patient_id": 3,
            "target": "no",
            "age": 53,
            "sex": "man",
            "sof": "yes",
            "diabetes": "no",
            "tenesmus": "no",
            "previous_rt": "no",
            "rectorrhagia": "no",
            "alcohol": 0,
            "tobacco": 2,
            "intestinal_habit": 4,
            "digestive_family_history": "stomach issue",
        },
        {
            "patient_id": 4,
            "target": "yes",
            "age": 61,
            "sex": "woman",
            "sof": "no",
            "diabetes": "yes",
            "tenesmus": "yes",
            "previous_rt": "no",
            "rectorrhagia": "yes",
            "alcohol": 4,
            "tobacco": 1,
            "intestinal_habit": 5,
            "digestive_family_history": "yes unsure",
        },
    ]
    _write_csv(data_csv, rows, ["patient_id", "target", *feature_columns])

    with questions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Variable", "Pregunta dirigida al paciente"])
        writer.writeheader()
        writer.writerows(
            [
                {"Variable": "sex", "Pregunta dirigida al paciente": "Sexo"},
                {"Variable": "sof", "Pregunta dirigida al paciente": "SOF"},
                {"Variable": "diabetes", "Pregunta dirigida al paciente": "Diabetes"},
                {"Variable": "tenesmus", "Pregunta dirigida al paciente": "Tenesmo"},
                {"Variable": "previous_rt", "Pregunta dirigida al paciente": "RT previa"},
                {"Variable": "rectorrhagia", "Pregunta dirigida al paciente": "Rectorragia"},
                {"Variable": "alcohol", "Pregunta dirigida al paciente": "Alcohol"},
                {"Variable": "tobacco", "Pregunta dirigida al paciente": "Tabaco"},
                {"Variable": "intestinal_habit", "Pregunta dirigida al paciente": "Habito intestinal"},
                {"Variable": "digestive_family_history", "Pregunta dirigida al paciente": "Historia familiar"},
            ]
        )

    return {
        "project": {"random_seed": 42},
        "dataset": {
            "data_csv": str(data_csv),
            "questions_csv": str(questions_csv),
            "csv_separator": ";",
            "csv_encoding": "latin1",
            "question_encoding": "utf-8",
            "id_column": "patient_id",
            "target_column": "target",
            "feature_columns": feature_columns,
            "binary_columns": ["sof", "diabetes", "tenesmus", "previous_rt", "rectorrhagia"],
            "ordinal_columns": ["alcohol", "tobacco", "intestinal_habit"],
            "target_mapping": {"no": 0, "yes": 1},
            "train_ratio": 0.5,
            "val_ratio": 0.25,
            "test_ratio": 0.25,
        },
    }


def test_normalize_family_history_handles_expected_buckets(tabular_module):
    assert tabular_module.normalize_family_history("Colon cancer") == "colon"
    assert tabular_module.normalize_family_history("stomach issue") == "gastric"
    assert tabular_module.normalize_family_history("yes maybe") == "other_positive"
    assert tabular_module.normalize_family_history("") == "unknown_dirty"


def test_prepare_dataframe_and_build_audit(tabular_module, sample_config):
    artifacts = tabular_module.prepare_tabular_dataframe(sample_config)
    report = tabular_module.build_tabular_audit_report(sample_config, artifacts)

    assert len(artifacts.dataframe) == 4
    assert "digestive_family_history_colon" in artifacts.feature_columns
    assert set(artifacts.dataframe["target"].unique()) == {0, 1}
    assert report["rows"] == 4
    assert report["target_distribution"]["riesgo_clinico"] == 2


def test_split_dataframe_adds_named_split(tabular_module, sample_config):
    artifacts = tabular_module.prepare_tabular_dataframe(sample_config)
    expanded = pd.concat([artifacts.dataframe, artifacts.dataframe], ignore_index=True)
    expanded["row_index"] = range(len(expanded))
    split_df = tabular_module.split_dataframe(expanded, sample_config)

    assert set(split_df["split"].unique()) == {"train", "val", "test"}
    assert len(split_df) == len(expanded)


def test_build_question_specs_adds_age_when_missing(tabular_module):
    config = {"dataset": {"target_column": "target", "binary_columns": [],}}
    questions = pd.DataFrame(
        [{"Variable": "sex", "Pregunta dirigida al paciente": "Sexo"}]
    )

    specs = tabular_module.build_question_specs(config, questions)

    assert specs[0]["variable"] == "age"
    assert specs[1]["variable"] == "sex"


def test_transform_patient_payload_encodes_family_history(tabular_module):
    signature = {
        "encoded_feature_columns": [
            "age",
            "sex",
            "sof",
            "alcohol",
            "digestive_family_history_no",
            "digestive_family_history_colon",
            "digestive_family_history_gastric",
            "digestive_family_history_other_positive",
            "digestive_family_history_unknown_dirty",
        ],
        "binary_columns": ["sof"],
        "ordinal_columns": ["alcohol"],
        "family_history_categories": ["no", "colon", "gastric", "other_positive", "unknown_dirty"],
    }
    payload = {"age": 55, "sex": 1, "sof": 0, "alcohol": 2, "digestive_family_history": "colon"}

    frame = tabular_module.transform_patient_payload(payload, signature)

    assert frame.iloc[0]["digestive_family_history_colon"] == 1
    assert frame.iloc[0]["digestive_family_history_no"] == 0


def test_transform_patient_payload_rejects_missing_fields(tabular_module):
    signature = {
        "encoded_feature_columns": ["age", "sex"],
        "binary_columns": [],
        "ordinal_columns": [],
        "family_history_categories": ["no", "colon", "gastric", "other_positive", "unknown_dirty"],
    }

    with pytest.raises(ValueError):
        tabular_module.transform_patient_payload({"age": 40}, signature)
