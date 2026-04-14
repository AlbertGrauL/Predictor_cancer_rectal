from __future__ import annotations

import argparse
import json
import pickle
import random
from typing import Any

import pandas as pd

from .config import load_config
from .metrics import compute_classification_metrics
from .utils import dependency_guard, load_paths, resolve_path, set_seed, to_project_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un modelo tabular de riesgo clínico.")
    parser.add_argument("--config", default="Predictor_models/configs/tabular_baseline.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--manifest", default="Predictor_models/artifacts/tabular/manifests/tabular_manifest.csv")
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    return parser.parse_args()


def sample_frame(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if not limit or len(frame) <= limit:
        return frame
    sampled_indices = list(frame.index)
    random.Random(seed).shuffle(sampled_indices)
    return frame.loc[sampled_indices[:limit]].copy()


def feature_columns_from_manifest(manifest: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    return [
        column
        for column in manifest.columns
        if (column.startswith("digestive_family_history_") and column != "digestive_family_history_group")
        or column in {"age", "sex", *config["dataset"]["binary_columns"], *config["dataset"]["ordinal_columns"]}
    ]


def build_tabular_estimator(model_name: str, config: dict[str, Any], y_train: pd.Series):
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        params = {key: value for key, value in config["models"]["random_forest"].items() if key != "search_space"}
        params.setdefault("random_state", config["project"]["random_seed"])
        return RandomForestClassifier(**params)

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        params = {key: value for key, value in config["models"]["xgboost"].items() if key != "search_space"}
        positive = int((y_train == 1).sum())
        negative = int((y_train == 0).sum())
        params["scale_pos_weight"] = negative / max(positive, 1)
        params.setdefault("objective", "binary:logistic")
        params.setdefault("eval_metric", "logloss")
        params.setdefault("random_state", config["project"]["random_seed"])
        return XGBClassifier(**params)

    raise ValueError(f"Modelo tabular no soportado: {model_name}")


def build_search_space(model_name: str, config: dict[str, Any], y_train: pd.Series) -> dict[str, list[Any]]:
    search_space = dict(config["models"][model_name].get("search_space", {}))
    if model_name == "xgboost":
        positive = int((y_train == 1).sum())
        negative = int((y_train == 0).sum())
        search_space["scale_pos_weight"] = [negative / max(positive, 1)]
    return search_space


def make_search(model_name: str, estimator, config: dict[str, Any], y_train: pd.Series):
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    cv = StratifiedKFold(
        n_splits=int(config["training"].get("cv_folds", 5)),
        shuffle=True,
        random_state=config["project"]["random_seed"],
    )
    return RandomizedSearchCV(
        estimator=estimator,
        param_distributions=build_search_space(model_name, config, y_train),
        n_iter=int(config["training"].get("search_iterations", 10)),
        scoring=str(config["training"].get("search_scoring", "f1")),
        refit=str(config["training"].get("refit_scoring", config["training"].get("search_scoring", "f1"))),
        cv=cv,
        random_state=config["project"]["random_seed"],
        n_jobs=1,
        verbose=0,
    )


def calibrate_estimator(estimator, X_val: pd.DataFrame, y_val: pd.Series, config: dict[str, Any]):
    if not config.get("calibration", {}).get("enabled", True):
        return estimator, {"enabled": False, "method": None}

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(estimator),
        method=str(config["calibration"].get("method", "sigmoid")),
        cv=None,
    )
    calibrator.fit(X_val, y_val)
    return calibrator, {"enabled": True, "method": str(config["calibration"].get("method", "sigmoid"))}


def evaluate_split(model, features: pd.DataFrame, target: pd.Series, class_names: list[str], positive_index: int) -> dict:
    probabilities = model.predict_proba(features)
    predictions = model.predict(features)
    return compute_classification_metrics(
        y_true=target.tolist(),
        y_pred=predictions.tolist(),
        y_score=probabilities.tolist(),
        class_names=class_names,
        positive_index=positive_index,
    )


def summarize_cv_results(search) -> list[dict[str, Any]]:
    results = []
    cv_results = search.cv_results_
    indices = sorted(
        range(len(cv_results["mean_test_score"])),
        key=lambda index: cv_results["rank_test_score"][index],
    )[:5]
    for index in indices:
        params = cv_results["params"][index]
        results.append(
            {
                "rank": int(cv_results["rank_test_score"][index]),
                "mean_test_score": float(cv_results["mean_test_score"][index]),
                "std_test_score": float(cv_results["std_test_score"][index]),
                "params": params,
            }
        )
    return results


def main() -> None:
    dependency_guard(
        {
            "numpy": "numpy",
            "pandas": "pandas",
            "yaml": "pyyaml",
            "sklearn": "scikit-learn",
            "xgboost": "xgboost",
        }
    )
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)
    set_seed(config["project"]["random_seed"])

    model_name = args.model or config["models"]["baseline"]
    manifest = pd.read_csv(resolve_path(args.manifest))
    feature_columns = feature_columns_from_manifest(manifest, config)
    target_column = config["dataset"]["target_column"]
    class_names = ["sin_riesgo_clinico", "riesgo_clinico"]
    positive_index = 1

    train_df = sample_frame(
        manifest[manifest["split"] == "train"],
        args.max_samples_per_split,
        config["project"]["random_seed"],
    )
    val_df = sample_frame(manifest[manifest["split"] == "val"], args.max_samples_per_split, config["project"]["random_seed"] + 1)

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_val = val_df[feature_columns]
    y_val = val_df[target_column]

    base_estimator = build_tabular_estimator(model_name, config, y_train)
    search = make_search(model_name, base_estimator, config, y_train)
    search.fit(X_train, y_train)

    best_estimator = search.best_estimator_
    calibrated_model, calibration_info = calibrate_estimator(best_estimator, X_val, y_val, config)
    val_metrics = evaluate_split(calibrated_model, X_val, y_val, class_names, positive_index)

    checkpoint_path = paths.checkpoints_dir / f"{model_name}_tabular.pkl"
    preprocessing_signature = {
        "encoded_feature_columns": feature_columns,
        "binary_columns": list(config["dataset"]["binary_columns"]),
        "ordinal_columns": list(config["dataset"]["ordinal_columns"]),
        "family_history_categories": ["no", "colon", "gastric", "other_positive", "unknown_dirty"],
    }
    payload = {
        "model": calibrated_model,
        "base_model": best_estimator,
        "model_name": model_name,
        "modality": "tabular",
        "class_names": class_names,
        "target_name": target_column,
        "feature_columns": feature_columns,
        "preprocessing_signature": preprocessing_signature,
        "config": config,
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "cv_results_summary": summarize_cv_results(search),
        "calibration": calibration_info,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("wb") as handle:
        pickle.dump(payload, handle)

    training_setup = {
        "model_hyperparameters": config["models"][model_name],
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "cv_folds": int(config["training"].get("cv_folds", 5)),
        "search_iterations": int(config["training"].get("search_iterations", 10)),
        "search_scoring": str(config["training"].get("search_scoring", "f1")),
        "calibration": calibration_info,
        "dataset": {"target_column": target_column, "feature_columns": feature_columns},
    }

    metadata_path = paths.metrics_dir / f"{model_name}_metadata.json"
    history_path = paths.metrics_dir / f"{model_name}_history.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "modality": "tabular",
                "class_names": class_names,
                "feature_columns": feature_columns,
                "target_name": target_column,
                "checkpoint_path": to_project_relative(checkpoint_path),
                "preprocessing_signature": preprocessing_signature,
                "training_setup": training_setup,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    history_path.write_text(
        json.dumps(
            [
                {
                    "stage": "cross_validation_search",
                    "cv_best_score": float(search.best_score_),
                    "best_params": search.best_params_,
                },
                {
                    "stage": "validation_calibrated",
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision_positive": val_metrics.get("precision_positive"),
                    "val_recall_positive": val_metrics.get("recall_positive"),
                    "val_f1_positive": val_metrics.get("f1_positive"),
                    "val_roc_auc": val_metrics.get("roc_auc"),
                    "val_pr_auc": val_metrics.get("pr_auc"),
                },
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Modelo tabular guardado en: {to_project_relative(checkpoint_path)}")
    print(f"Mejores parámetros: {json.dumps(search.best_params_, ensure_ascii=False)}")
    print(f"Mejor score CV: {search.best_score_:.4f}")
    print(f"Metadatos guardados en: {to_project_relative(metadata_path)}")
    print(f"Historial guardado en: {to_project_relative(history_path)}")


if __name__ == "__main__":
    main()
