from __future__ import annotations

import argparse
import csv
import json

from ..config import load_config
from ..utils import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume experimentos y comparativas de modelos.")
    parser.add_argument("--config", default="Predictor_models/configs/image/multiclass_baseline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)
    expected_class_count = len(config["dataset"]["classes"])
    rows = []
    per_class_rows = []

    for evaluation_path in sorted(paths.metrics_dir.glob("*_evaluation.json")):
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        metadata_path = paths.metrics_dir / f"{payload['model_name']}_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        class_count = len(payload.get("class_names", metrics.get("class_names", metadata.get("class_names", []))))
        if class_count != expected_class_count:
            continue
        rows.append(
            {
                "model_name": payload["model_name"],
                "class_count": class_count,
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics.get("precision_macro", metrics.get("precision")),
                "recall_macro": metrics.get("recall_macro", metrics.get("recall")),
                "f1_macro": metrics.get("f1_macro", metrics.get("f1")),
                "f1_weighted": metrics.get("f1_weighted", metrics.get("f1")),
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "hard_cases": len(payload.get("hard_cases", [])),
            }
        )
        for class_name, class_metrics in metrics.get("per_class", {}).items():
            per_class_rows.append(
                {
                    "model_name": payload["model_name"],
                    "class_name": class_name,
                    "precision": class_metrics.get("precision"),
                    "recall": class_metrics.get("recall"),
                    "f1": class_metrics.get("f1"),
                    "support": class_metrics.get("support"),
                }
            )

    output_path = paths.reports_dir / "experiment_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "class_count",
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "f1_weighted",
                "roc_auc",
                "pr_auc",
                "hard_cases",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    per_class_output_path = paths.reports_dir / "experiment_summary_per_class.csv"
    with per_class_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_name", "class_name", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        writer.writerows(per_class_rows)

    print(f"Resumen exportado en: {output_path}")
    print(f"Resumen por clase exportado en: {per_class_output_path}")


if __name__ == "__main__":
    main()
