from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _package_path(dotted_name: str) -> Path | None:
    relative = Path(*dotted_name.split("."))
    candidate = ROOT / relative
    if candidate.is_dir():
        return candidate
    return None


def ensure_package(monkeypatch, dotted_name: str):
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]

    module = types.ModuleType(dotted_name)
    package_path = _package_path(dotted_name)
    module.__path__ = [str(package_path)] if package_path else []
    monkeypatch.setitem(sys.modules, dotted_name, module)

    if "." in dotted_name:
        parent_name, _, child_name = dotted_name.rpartition(".")
        parent = ensure_package(monkeypatch, parent_name)
        setattr(parent, child_name, module)

    return module


def make_module(monkeypatch, dotted_name: str, **attributes):
    if "." in dotted_name:
        parent_name, _, child_name = dotted_name.rpartition(".")
        parent = ensure_package(monkeypatch, parent_name)
    else:
        parent = None
        child_name = dotted_name

    module = types.ModuleType(dotted_name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, dotted_name, module)

    if parent is not None:
        setattr(parent, child_name, module)

    return module


def import_fresh(monkeypatch, dotted_name: str):
    monkeypatch.delitem(sys.modules, dotted_name, raising=False)
    importlib.invalidate_caches()
    return importlib.import_module(dotted_name)
