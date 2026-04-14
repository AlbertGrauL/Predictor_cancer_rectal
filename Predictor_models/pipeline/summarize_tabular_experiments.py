from __future__ import annotations

import argparse
import csv
import json

from .config import load_config
from .utils import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume experimentos de modelos tabulares.")
    parser.add_argument("--config", default="Predictor_models/configs/tabular_baseline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)

    rows = []
    for evaluation_path in sorted(paths.metrics_dir.glob("*_evaluation.json")):
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if payload.get("modality") != "tabular":
            continue
        metrics = payload["metrics"]
        rows.append(
            {
                "model_name": payload["model_name"],
                "accuracy": metrics["accuracy"],
                "precision_positive": metrics.get("precision_positive"),
                "recall_positive": metrics.get("recall_positive"),
                "f1_positive": metrics.get("f1_positive"),
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "cv_best_score": payload.get("training_setup", {}).get("cv_best_score"),
                "calibration_method": (payload.get("training_setup", {}).get("calibration") or {}).get("method"),
            }
        )

    output_path = paths.reports_dir / "experiment_summary_tabular.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "accuracy",
                "precision_positive",
                "recall_positive",
                "f1_positive",
                "roc_auc",
                "pr_auc",
                "cv_best_score",
                "calibration_method",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resumen tabular exportado en: {output_path}")


if __name__ == "__main__":
    main()
