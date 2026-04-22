from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

from ..config import load_config
from ..utils import load_paths, set_seed, to_project_relative, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera manifiesto y splits reproducibles.")
    parser.add_argument("--config", default="Predictor_models/configs/image/multiclass_baseline.yaml")
    return parser.parse_args()


def discover_images(root: Path, extensions: set[str]) -> list[Path]:
    return [item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in extensions]


def hash_prefix(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(chunk_size))
    return digest.hexdigest()[:16]


def create_manifest(config: dict) -> list[dict[str, str | int]]:
    paths = load_paths(config)
    dataset = config["dataset"]
    extensions = {ext.lower() for ext in dataset["extensions"]}

    rows: list[dict[str, str | int]] = []
    for class_name, relative_sources in dataset["classes"].items():
        for relative_source in relative_sources:
            source_root = paths.dataset_root / relative_source
            source_name = relative_source.replace("\\", "/")
            for file_path in discover_images(source_root, extensions):
                rows.append(
                    {
                        "path": to_project_relative(file_path),
                        "class_name": class_name,
                        "source_name": source_name,
                        "file_name": file_path.name,
                        "group_id": f"{class_name}:{hash_prefix(file_path)}",
                    }
                )
    return rows


def stratified_split(rows: list[dict[str, str | int]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[int]]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        grouped[str(row["class_name"])][str(row["group_id"])].append(index)

    random.seed(seed)
    splits = {"train": [], "val": [], "test": []}
    for class_groups in grouped.values():
        group_items = list(class_groups.values())
        random.shuffle(group_items)
        total = len(group_items)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        for group_indices in group_items[:train_end]:
            splits["train"].extend(group_indices)
        for group_indices in group_items[train_end:val_end]:
            splits["val"].extend(group_indices)
        for group_indices in group_items[val_end:]:
            splits["test"].extend(group_indices)

    for split_name in splits:
        splits[split_name].sort()
    return splits


def attach_split(rows: list[dict[str, str | int]], split_map: dict[str, list[int]]) -> list[dict[str, str | int]]:
    index_to_split: dict[int, str] = {}
    for split_name, indices in split_map.items():
        for index in indices:
            index_to_split[index] = split_name

    enriched: list[dict[str, str | int]] = []
    for index, row in enumerate(rows):
        enriched_row = dict(row)
        enriched_row["split"] = index_to_split[index]
        enriched.append(enriched_row)
    return enriched


def write_manifest(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "class_name", "source_name", "file_name", "group_id", "split"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str | int]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        summary[str(row["split"])][str(row["class_name"])] += 1
    return {split: dict(values) for split, values in summary.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["project"]["random_seed"])
    paths = load_paths(config)
    dataset = config["dataset"]

    rows = create_manifest(config)
    split_map = stratified_split(
        rows=rows,
        train_ratio=dataset["train_ratio"],
        val_ratio=dataset["val_ratio"],
        seed=config["project"]["random_seed"],
    )
    enriched = attach_split(rows, split_map)

    manifest_path = paths.manifests_dir / "dataset_manifest.csv"
    splits_path = paths.manifests_dir / "splits.json"
    summary_path = paths.reports_dir / "split_summary.json"

    write_manifest(manifest_path, enriched)
    write_json(splits_path, split_map)
    write_json(summary_path, summarize(enriched))

    print(f"Manifest generado en: {to_project_relative(manifest_path)}")
    print(f"Splits generados en: {to_project_relative(splits_path)}")
    print(f"Resumen generado en: {to_project_relative(summary_path)}")


if __name__ == "__main__":
    main()
