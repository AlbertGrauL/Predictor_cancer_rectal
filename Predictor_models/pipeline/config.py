from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import resolve_path


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML no esta instalado. Ejecuta `uv sync`.") from exc

    resolved = resolve_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuracion invalida en {resolved}")
    return data
