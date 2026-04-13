from __future__ import annotations

import argparse
import shutil

from .config import load_config
from .utils import load_paths, to_project_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limpia artefactos generados del proyecto.")
    parser.add_argument("--config", default="Predictor_models/configs/multiclass_baseline.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)

    removed: list[str] = []
    for directory in [
        paths.checkpoints_dir,
        paths.figures_dir,
        paths.manifests_dir,
        paths.metrics_dir,
        paths.reports_dir,
    ]:
        for item in directory.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_file():
                item.unlink()
                removed.append(to_project_relative(item))
            elif item.is_dir():
                shutil.rmtree(item)
                removed.append(to_project_relative(item))

    if removed:
        print("Artefactos eliminados:")
        for item in removed:
            print(f"- {item}")
    else:
        print("No habia artefactos para eliminar.")


if __name__ == "__main__":
    main()
