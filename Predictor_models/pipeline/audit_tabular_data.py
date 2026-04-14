from __future__ import annotations

import argparse

from .config import load_config
from .tabular_utils import build_question_specs, build_tabular_audit_report, prepare_tabular_dataframe
from .utils import load_paths, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita el CSV tabular y el cuestionario de pacientes.")
    parser.add_argument("--config", default="Predictor_models/configs/tabular_baseline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)
    artifacts = prepare_tabular_dataframe(config)

    audit_report = build_tabular_audit_report(config, artifacts)
    question_specs = build_question_specs(config, artifacts.questions)

    audit_path = paths.reports_dir / "tabular_dataset_audit.json"
    questions_path = paths.reports_dir / "tabular_question_specs.json"
    write_json(audit_path, audit_report)
    write_json(questions_path, {"questions": question_specs})

    print(f"Auditoría tabular guardada en: {audit_path}")
    print(f"Especificación de preguntas guardada en: {questions_path}")


if __name__ == "__main__":
    main()
