from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from Predictor_models.pipeline.config import load_config
    from Predictor_models.pipeline.utils import resolve_path
else:
    from .config import load_config
    from .utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta la comparación tabular completa.")
    parser.add_argument("--config", default="Predictor_models/configs/tabular_baseline.yaml")
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def run_step(command: list[str], title: str, cwd: Path) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    print("Comando:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = resolve_path(".")
    models = args.models or config["models"]["candidates"]

    if not args.no_clean:
        run_step([sys.executable, "-m", "Predictor_models.pipeline.clean_artifacts", "--config", args.config], "0. Limpieza tabular", root)
    if not args.skip_audit:
        run_step([sys.executable, "-m", "Predictor_models.pipeline.audit_tabular_data", "--config", args.config], "1. Auditoría tabular", root)
    if not args.skip_prepare:
        run_step([sys.executable, "-m", "Predictor_models.pipeline.prepare_tabular_data", "--config", args.config], "2. Preparación tabular", root)

    for index, model_name in enumerate(models, start=1):
        checkpoint = root / "Predictor_models" / "artifacts" / "tabular" / "checkpoints" / f"{model_name}_tabular.pkl"
        run_step([sys.executable, "-m", "Predictor_models.pipeline.train_tabular", "--config", args.config, "--model", model_name], f"3.{index} Entrenamiento de {model_name}", root)
        run_step([sys.executable, "-m", "Predictor_models.pipeline.evaluate_tabular", "--config", args.config, "--checkpoint", str(checkpoint)], f"4.{index} Evaluación de {model_name}", root)

    run_step([sys.executable, "-m", "Predictor_models.pipeline.summarize_tabular_experiments", "--config", args.config], "5. Resumen tabular", root)
    print("\nComparación tabular completada correctamente.")


if __name__ == "__main__":
    main()
