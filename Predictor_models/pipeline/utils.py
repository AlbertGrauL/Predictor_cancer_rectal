from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    return ROOT


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def ensure_dir(path: str | Path) -> Path:
    resolved = resolve_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_json(path: str | Path) -> Any:
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return resolved


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def dependency_guard(modules: dict[str, str]) -> None:
    missing: list[str] = []
    for module_name, package_name in modules.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        raise RuntimeError(
            "Faltan dependencias requeridas: "
            + ", ".join(sorted(missing))
            + ". Instala el entorno con `uv sync`."
        )


@dataclass(slots=True)
class PathsConfig:
    dataset_root: Path
    artifacts_root: Path
    manifests_dir: Path
    reports_dir: Path
    figures_dir: Path
    checkpoints_dir: Path
    metrics_dir: Path


def load_paths(config: dict[str, Any]) -> PathsConfig:
    paths = config["paths"]
    return PathsConfig(
        dataset_root=resolve_path(paths["dataset_root"]),
        artifacts_root=ensure_dir(paths["artifacts_root"]),
        manifests_dir=ensure_dir(paths["manifests_dir"]),
        reports_dir=ensure_dir(paths["reports_dir"]),
        figures_dir=ensure_dir(paths["figures_dir"]),
        checkpoints_dir=ensure_dir(paths["checkpoints_dir"]),
        metrics_dir=ensure_dir(paths["metrics_dir"]),
    )
