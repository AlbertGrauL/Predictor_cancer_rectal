from __future__ import annotations

import argparse
import csv
import json

from .config import load_config
from .utils import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume experimentos y comparativas de modelos.")
    parser.add_argument("--config", default="Predictor_models/configs/binary_baseline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)
    rows = []

    for evaluation_path in sorted(paths.metrics_dir.glob("*_evaluation.json")):
        payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append(
            {
                "model_name": payload["model_name"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "hard_cases": len(payload.get("hard_cases", [])),
            }
        )

    output_path = paths.reports_dir / "experiment_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model_name", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "hard_cases"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Resumen exportado en: {output_path}")


if __name__ == "__main__":
    main()
