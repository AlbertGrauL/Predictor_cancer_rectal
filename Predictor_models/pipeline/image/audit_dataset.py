from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config import load_config
from ..utils import load_paths, set_seed, to_project_relative, write_json


@dataclass(slots=True)
class ImageInfo:
    path: Path
    class_name: str
    source_name: str
    width: int | None
    height: int | None
    image_type: str | None
    sha256_prefix: str
    is_corrupt: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audita el dataset del proyecto.")
    parser.add_argument("--config", default="Predictor_models/configs/image/multiclass_baseline.yaml")
    return parser.parse_args()


def detect_image_type(path: Path) -> str | None:
    with path.open("rb") as handle:
        header = handle.read(16)
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"\xff\xd8"):
        return "jpeg"
    return None


def jpg_dimensions(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as handle:
        if handle.read(2) != b"\xff\xd8":
            return None, None
        while True:
            marker_prefix = handle.read(1)
            if not marker_prefix:
                return None, None
            if marker_prefix != b"\xff":
                continue
            marker = handle.read(1)
            while marker == b"\xff":
                marker = handle.read(1)
            if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                _length = struct.unpack(">H", handle.read(2))[0]
                _precision = handle.read(1)
                height = struct.unpack(">H", handle.read(2))[0]
                width = struct.unpack(">H", handle.read(2))[0]
                return width, height
            if marker in {b"\xd8", b"\xd9"}:
                continue
            segment_length = struct.unpack(">H", handle.read(2))[0]
            handle.seek(segment_length - 2, 1)


def png_dimensions(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n":
        return None, None
    width, height = struct.unpack(">II", signature[16:24])
    return width, height


def infer_dimensions(path: Path) -> tuple[int | None, int | None, str | None]:
    image_type = detect_image_type(path)
    try:
        if image_type == "png":
            width, height = png_dimensions(path)
        elif image_type == "jpeg":
            width, height = jpg_dimensions(path)
        else:
            return None, None, image_type
        return width, height, image_type
    except Exception:
        return None, None, image_type


def hash_prefix(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
    return digest.hexdigest()[:16]


def iter_files(root: Path, allowed_exts: set[str]) -> Iterable[Path]:
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in allowed_exts:
            yield file_path


def collect_records(config: dict) -> list[ImageInfo]:
    paths = load_paths(config)
    dataset = config["dataset"]
    allowed_exts = {ext.lower() for ext in dataset["extensions"]}
    class_mapping = dataset["classes"]

    records: list[ImageInfo] = []
    for class_name, relative_sources in class_mapping.items():
        for relative_source in relative_sources:
            source_root = paths.dataset_root / relative_source
            source_name = relative_source.replace("\\", "/")
            for file_path in iter_files(source_root, allowed_exts):
                width, height, image_type = infer_dimensions(file_path)
                records.append(
                    ImageInfo(
                        path=file_path,
                        class_name=class_name,
                        source_name=source_name,
                        width=width,
                        height=height,
                        image_type=image_type,
                        sha256_prefix=hash_prefix(file_path),
                        is_corrupt=width is None or height is None or image_type is None,
                    )
                )
    return records


def summarize(records: list[ImageInfo], external_eval: dict[str, list[str]], dataset_root: Path) -> dict:
    class_counts = Counter(record.class_name for record in records)
    source_counts = Counter(record.source_name for record in records)
    type_counts = Counter(record.image_type or "unknown" for record in records)
    corrupt_files = [to_project_relative(record.path) for record in records if record.is_corrupt]

    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        duplicate_groups[record.sha256_prefix].append(to_project_relative(record.path))
    duplicates = [paths for paths in duplicate_groups.values() if len(paths) > 1]

    widths = [record.width for record in records if record.width]
    heights = [record.height for record in records if record.height]

    external_sources: dict[str, dict[str, int]] = {}
    for group_name, relative_paths in external_eval.items():
        group_counts: dict[str, int] = {}
        for relative_path in relative_paths:
            target = dataset_root / relative_path
            count = sum(1 for item in target.rglob("*") if item.is_file())
            group_counts[relative_path] = count
        external_sources[group_name] = group_counts

    return {
        "total_images": len(records),
        "class_counts": dict(class_counts),
        "source_counts": dict(source_counts),
        "format_counts": dict(type_counts),
        "resolution_summary": {
            "min_width": min(widths) if widths else None,
            "max_width": max(widths) if widths else None,
            "min_height": min(heights) if heights else None,
            "max_height": max(heights) if heights else None,
        },
        "corrupt_files": corrupt_files,
        "possible_duplicate_groups": duplicates,
        "duplicate_group_count": len(duplicates),
        "external_eval_sources": external_sources,
        "notes": [
            "El analisis de duplicados usa un hash parcial para localizar posibles coincidencias exactas.",
            "Los archivos marcados como corruptos deben revisarse antes del entrenamiento definitivo.",
            "Si hay fuentes muy parecidas visualmente, conviene analizar fuga de datos por subdataset.",
        ],
    }


def write_markdown(report_path: Path, summary: dict) -> None:
    lines = [
        "# Informe de auditoria del dataset",
        "",
        f"- Total de imagenes auditadas: **{summary['total_images']}**",
        f"- Posibles grupos de duplicados: **{summary['duplicate_group_count']}**",
        f"- Archivos potencialmente corruptos: **{len(summary['corrupt_files'])}**",
        "",
        "## Conteo por clase",
        "",
    ]
    for class_name, count in summary["class_counts"].items():
        lines.append(f"- `{class_name}`: {count}")
    lines.extend(["", "## Conteo por fuente", ""])
    for source_name, count in summary["source_counts"].items():
        lines.append(f"- `{source_name}`: {count}")
    lines.extend(["", "## Formatos detectados", ""])
    for image_type, count in summary["format_counts"].items():
        lines.append(f"- `{image_type}`: {count}")
    lines.extend(["", "## Fuentes reservadas para evaluacion externa", ""])
    for group_name, values in summary["external_eval_sources"].items():
        lines.append(f"- `{group_name}`")
        for relative_path, count in values.items():
            lines.append(f"  - `{relative_path}`: {count}")
    lines.extend(["", "## Notas", ""])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["project"]["random_seed"])
    paths = load_paths(config)

    records = collect_records(config)
    summary = summarize(records, config["dataset"].get("external_eval_sources", {}), paths.dataset_root)
    summary["config_path"] = args.config

    json_path = paths.reports_dir / "dataset_audit.json"
    md_path = paths.reports_dir / "dataset_audit.md"
    write_json(json_path, summary)
    write_markdown(md_path, summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nInforme JSON: {to_project_relative(json_path)}")
    print(f"Informe Markdown: {to_project_relative(md_path)}")


if __name__ == "__main__":
    main()
