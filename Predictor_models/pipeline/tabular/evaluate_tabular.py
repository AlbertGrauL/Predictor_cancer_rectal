from __future__ import annotations

import argparse
from collections import Counter
import pickle

import pandas as pd

from ..config import load_config
from ..metrics import compute_classification_metrics, save_curves
from ..utils import dependency_guard, load_paths, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalúa un modelo tabular.")
    parser.add_argument("--config", default="Predictor_models/configs/tabular/tabular_baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="Predictor_models/artifacts/tabular/manifests/tabular_manifest.csv")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def summarize_feature_importance(model, feature_columns: list[str]) -> list[dict[str, float | str]]:
    if not hasattr(model, "feature_importances_"):
        return []
    ranking = sorted(
        zip(feature_columns, model.feature_importances_.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"feature": feature, "importance": float(importance)} for feature, importance in ranking[:10]]


def summarize_permutation_importance(model, features: pd.DataFrame, target: pd.Series, feature_columns: list[str]) -> list[dict[str, float | str]]:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, features, target, n_repeats=8, random_state=42, scoring="f1")
    ranking = sorted(
        zip(feature_columns, result.importances_mean.tolist()),
        key=lambda item: item[1],
        reverse=True,
    )
    return [{"feature": feature, "importance": float(importance)} for feature, importance in ranking[:10]]


def build_group_breakdown(test_df: pd.DataFrame, predictions: list[int], target_column: str) -> dict[str, dict[str, float | int]]:
    enriched = test_df.copy()
    enriched["prediction"] = predictions
    enriched["error"] = (enriched["prediction"] != enriched[target_column]).astype(int)
    breakdown = {}
    for group_name, group_df in enriched.groupby("digestive_family_history_group"):
        breakdown[str(group_name)] = {
            "total": int(len(group_df)),
            "errors": int(group_df["error"].sum()),
            "error_rate": float(group_df["error"].mean()),
        }
    return breakdown


def build_source_alerts(breakdown: dict[str, dict[str, float | int]]) -> list[dict[str, float | int | str]]:
    valid_rates = [(name, values["error_rate"], values["total"]) for name, values in breakdown.items() if values["total"] >= 25]
    if len(valid_rates) < 2:
        return []
    average_rate = sum(rate for _name, rate, _count in valid_rates) / len(valid_rates)
    alerts = []
    for name, rate, total in valid_rates:
        if rate >= average_rate * 1.5:
            alerts.append({"source_name": name, "alert": "high_error_rate", "error_rate": rate, "samples": total})
        elif rate <= max(0.01, average_rate * 0.5):
            alerts.append({"source_name": name, "alert": "very_low_error_rate", "error_rate": rate, "samples": total})
    return alerts


def main() -> None:
    dependency_guard(
        {
            "numpy": "numpy",
            "pandas": "pandas",
            "yaml": "pyyaml",
            "sklearn": "scikit-learn",
            "matplotlib": "matplotlib",
            "seaborn": "seaborn",
            "xgboost": "xgboost",
        }
    )
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)
    manifest = pd.read_csv(resolve_path(args.manifest))
    test_df = manifest[manifest["split"] == "test"].copy()
    if args.max_samples and len(test_df) > args.max_samples:
        test_df = test_df.sample(n=args.max_samples, random_state=config["project"]["random_seed"]).copy()

    with resolve_path(args.checkpoint).open("rb") as handle:
        payload = pickle.load(handle)

    calibrated_model = payload["model"]
    base_model = payload.get("base_model", calibrated_model)
    feature_columns = payload["feature_columns"]
    class_names = payload["class_names"]
    model_name = payload["model_name"]
    target_column = payload["target_name"]

    probabilities = calibrated_model.predict_proba(test_df[feature_columns])
    predictions = calibrated_model.predict(test_df[feature_columns])
    metrics = compute_classification_metrics(
        y_true=test_df[target_column].tolist(),
        y_pred=predictions.tolist(),
        y_score=probabilities.tolist(),
        class_names=class_names,
        positive_index=1,
    )
    curve_paths = save_curves(
        y_true=test_df[target_column].tolist(),
        y_score=probabilities.tolist(),
        figures_dir=paths.figures_dir,
        prefix=model_name,
        class_names=class_names,
    )

    confusion_pairs = Counter()
    for truth, pred in zip(test_df[target_column].tolist(), predictions.tolist()):
        if truth != pred:
            confusion_pairs[(class_names[truth], class_names[pred])] += 1

    breakdown = build_group_breakdown(test_df, predictions.tolist(), target_column)
    report = {
        "model_name": model_name,
        "modality": "tabular",
        "class_names": class_names,
        "metrics": metrics,
        "curve_paths": curve_paths,
        "preprocessing_signature": payload["preprocessing_signature"],
        "training_setup": {
            "model_hyperparameters": config["models"][model_name],
            "best_params": payload.get("best_params", {}),
            "cv_best_score": payload.get("cv_best_score"),
            "calibration": payload.get("calibration", {"enabled": False, "method": None}),
        },
        "feature_importance": summarize_feature_importance(base_model, feature_columns),
        "permutation_importance": summarize_permutation_importance(
            calibrated_model,
            test_df[feature_columns],
            test_df[target_column],
            feature_columns,
        ),
        "cv_results_summary": payload.get("cv_results_summary", []),
        "confusion_summary": [
            {"true_class": truth, "predicted_class": pred, "count": count}
            for (truth, pred), count in confusion_pairs.most_common(10)
        ],
        "by_source": breakdown,
        "source_alerts": build_source_alerts(breakdown),
    }
    report_path = paths.metrics_dir / f"{model_name}_evaluation.json"
    write_json(report_path, report)
    print(f"Evaluación tabular guardada en: {report_path}")


if __name__ == "__main__":
    main()
