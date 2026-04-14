from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
import unicodedata
from typing import Any

import pandas as pd

from .utils import resolve_path


FAMILY_HISTORY_CATEGORIES = ["no", "colon", "gastric", "other_positive", "unknown_dirty"]


@dataclass(slots=True)
class TabularArtifacts:
    dataframe: pd.DataFrame
    questions: pd.DataFrame
    preprocessing_signature: dict[str, Any]
    feature_columns: list[str]
    class_names: list[str]
    positive_index: int


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_family_history(value: Any) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return "unknown_dirty"
    if normalized == "no":
        return "no"
    if "colon" in normalized:
        return "colon"
    if "gastric" in normalized or "stomach" in normalized:
        return "gastric"
    if normalized.startswith("yes") or normalized in {"pak", "unesco", "anca", "anque"}:
        return "other_positive"
    return "unknown_dirty"


def load_questions(config: dict[str, Any]) -> pd.DataFrame:
    question_path = resolve_path(config["dataset"]["questions_csv"])
    try:
        questions = pd.read_csv(question_path, encoding=config["dataset"].get("question_encoding", "utf-8"))
    except UnicodeDecodeError:
        questions = pd.read_csv(question_path, encoding="latin1")
    questions.columns = [str(column).strip() for column in questions.columns]
    return questions


def load_raw_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    dataset_cfg = config["dataset"]
    dataframe = pd.read_csv(
        resolve_path(dataset_cfg["data_csv"]),
        sep=dataset_cfg.get("csv_separator", ";"),
        encoding=dataset_cfg.get("csv_encoding", "latin1"),
    )
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    return dataframe


def prepare_tabular_dataframe(config: dict[str, Any]) -> TabularArtifacts:
    dataset_cfg = config["dataset"]
    dataframe = load_raw_dataframe(config)
    questions = load_questions(config)

    target_column = dataset_cfg["target_column"]
    required_columns = [dataset_cfg["id_column"], target_column, *dataset_cfg["feature_columns"]]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas en el CSV tabular: {missing_columns}")

    if "Variable" not in questions.columns or "Pregunta dirigida al paciente" not in questions.columns:
        raise ValueError("El CSV de preguntas debe contener 'Variable' y 'Pregunta dirigida al paciente'.")

    prepared = dataframe.copy()
    prepared["row_index"] = range(len(prepared))
    prepared[target_column] = prepared[target_column].astype(str).str.strip().str.lower()
    prepared[target_column] = prepared[target_column].map(dataset_cfg["target_mapping"])
    if prepared[target_column].isna().any():
        invalid = dataframe.loc[prepared[target_column].isna(), target_column].astype(str).unique().tolist()
        raise ValueError(f"Valores desconocidos en la columna objetivo: {invalid}")

    prepared["sex"] = prepared["sex"].astype(str).str.strip().str.lower().map({"woman": 0, "man": 1})
    if prepared["sex"].isna().any():
        invalid = dataframe.loc[prepared["sex"].isna(), "sex"].astype(str).unique().tolist()
        raise ValueError(f"Valores desconocidos en sex: {invalid}")

    unexpected_values: dict[str, list[str]] = {}
    for column in dataset_cfg["binary_columns"]:
        normalized = prepared[column].astype(str).str.strip().str.lower()
        invalid_mask = ~normalized.isin({"yes", "no"})
        if invalid_mask.any():
            unexpected_values[column] = sorted(normalized[invalid_mask].unique().tolist())
        prepared[column] = normalized.map({"yes": 1, "no": 0})

    prepared["digestive_family_history_group"] = prepared["digestive_family_history"].apply(normalize_family_history)
    family_dummies = pd.get_dummies(
        pd.Categorical(prepared["digestive_family_history_group"], categories=FAMILY_HISTORY_CATEGORIES),
        prefix="digestive_family_history",
        dtype=int,
    )
    prepared = pd.concat([prepared, family_dummies], axis=1)

    numeric_columns = ["age", "sex", *dataset_cfg["binary_columns"], *dataset_cfg["ordinal_columns"]]
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        if prepared[column].isna().any():
            unexpected_values.setdefault(column, []).extend(
                dataframe.loc[prepared[column].isna(), column].astype(str).unique().tolist()
            )

    feature_columns = [
        "age",
        "sex",
        *dataset_cfg["binary_columns"],
        *dataset_cfg["ordinal_columns"],
        *family_dummies.columns.tolist(),
    ]
    preprocessing_signature = {
        "target_column": target_column,
        "id_column": dataset_cfg["id_column"],
        "raw_feature_columns": list(dataset_cfg["feature_columns"]),
        "encoded_feature_columns": feature_columns,
        "binary_columns": list(dataset_cfg["binary_columns"]),
        "ordinal_columns": list(dataset_cfg["ordinal_columns"]),
        "family_history_categories": list(FAMILY_HISTORY_CATEGORIES),
        "sex_mapping": {"woman": 0, "man": 1},
        "unexpected_values": {key: sorted(set(values)) for key, values in unexpected_values.items()},
    }

    return TabularArtifacts(
        dataframe=prepared,
        questions=questions,
        preprocessing_signature=preprocessing_signature,
        feature_columns=feature_columns,
        class_names=["sin_riesgo_clinico", "riesgo_clinico"],
        positive_index=1,
    )


def split_dataframe(dataframe: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    dataset_cfg = config["dataset"]
    target_column = dataset_cfg["target_column"]
    seed = config["project"]["random_seed"]

    train_val_idx, test_idx = train_test_split(
        dataframe.index,
        test_size=dataset_cfg["test_ratio"],
        random_state=seed,
        stratify=dataframe[target_column],
    )
    val_relative = dataset_cfg["val_ratio"] / (dataset_cfg["train_ratio"] + dataset_cfg["val_ratio"])
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative,
        random_state=seed,
        stratify=dataframe.loc[train_val_idx, target_column],
    )

    split_map = {index: "train" for index in train_idx}
    split_map.update({index: "val" for index in val_idx})
    split_map.update({index: "test" for index in test_idx})

    output = dataframe.copy()
    output["split"] = output.index.map(split_map)
    return output


def build_tabular_audit_report(config: dict[str, Any], artifacts: TabularArtifacts) -> dict[str, Any]:
    dataframe = artifacts.dataframe
    questions = artifacts.questions
    target_column = config["dataset"]["target_column"]
    question_variables = [str(value).strip() for value in questions["Variable"].tolist()]
    missing_question_features = [
        column for column in config["dataset"]["feature_columns"] if column not in question_variables
    ]
    extra_question_variables = [
        column
        for column in question_variables
        if column not in config["dataset"]["feature_columns"] and column != target_column
    ]

    family_counts = Counter(dataframe["digestive_family_history_group"].astype(str).tolist())
    return {
        "rows": int(len(dataframe)),
        "target_distribution": {
            "sin_riesgo_clinico": int((dataframe[target_column] == 0).sum()),
            "riesgo_clinico": int((dataframe[target_column] == 1).sum()),
        },
        "feature_columns": artifacts.feature_columns,
        "unexpected_values": artifacts.preprocessing_signature["unexpected_values"],
        "digestive_family_history_group_distribution": dict(family_counts),
        "missing_question_features": missing_question_features,
        "extra_question_variables": extra_question_variables,
    }


def build_question_specs(config: dict[str, Any], questions: pd.DataFrame) -> list[dict[str, Any]]:
    target_column = config["dataset"]["target_column"]
    specs: list[dict[str, Any]] = []
    seen_variables: set[str] = set()
    for row in questions.to_dict(orient="records"):
        variable = str(row["Variable"]).strip()
        if variable == target_column:
            continue
        seen_variables.add(variable)
        question_text = str(row["Pregunta dirigida al paciente"]).strip()
        if variable == "sex":
            options = [{"label": "Mujer", "value": 0}, {"label": "Hombre", "value": 1}]
        elif variable in config["dataset"]["binary_columns"]:
            options = [{"label": "No", "value": 0}, {"label": "Sí", "value": 1}]
        elif variable == "tobacco":
            options = [{"label": f"Nivel {value}", "value": value} for value in [0, 1, 2]]
        elif variable == "alcohol":
            options = [{"label": f"Nivel {value}", "value": value} for value in [0, 1, 2, 3, 4]]
        elif variable == "intestinal_habit":
            options = [{"label": f"Categoría {value}", "value": value} for value in [0, 1, 2, 3, 4, 5]]
        elif variable == "digestive_family_history":
            options = [
                {"label": "No", "value": "no"},
                {"label": "Colon", "value": "colon"},
                {"label": "Gástrico", "value": "gastric"},
                {"label": "Otro positivo", "value": "other_positive"},
                {"label": "No seguro / otro", "value": "unknown_dirty"},
            ]
        elif variable == "age":
            options = []
        else:
            continue
        specs.append({"variable": variable, "question": question_text, "options": options})
    if "age" not in seen_variables:
        specs.insert(0, {"variable": "age", "question": "¿Cuál es la edad del paciente?", "options": []})
    return specs


def transform_patient_payload(payload: dict[str, Any], preprocessing_signature: dict[str, Any]) -> pd.DataFrame:
    required_columns = [
        "age",
        "sex",
        *preprocessing_signature["binary_columns"],
        *preprocessing_signature["ordinal_columns"],
        "digestive_family_history",
    ]
    missing = [column for column in required_columns if column not in payload]
    if missing:
        raise ValueError(f"Faltan campos del formulario tabular: {missing}")

    row = {
        "age": int(payload["age"]),
        "sex": int(payload["sex"]),
    }
    for column in preprocessing_signature["binary_columns"]:
        row[column] = int(payload[column])
    for column in preprocessing_signature["ordinal_columns"]:
        row[column] = int(payload[column])

    selected_group = str(payload["digestive_family_history"])
    if selected_group not in preprocessing_signature["family_history_categories"]:
        selected_group = "unknown_dirty"
    for category in preprocessing_signature["family_history_categories"]:
        row[f"digestive_family_history_{category}"] = 1 if category == selected_group else 0

    ordered = {column: row.get(column, 0) for column in preprocessing_signature["encoded_feature_columns"]}
    return pd.DataFrame([ordered])
