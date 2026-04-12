from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from .utils import resolve_path


def load_manifest(path: str | Path, split: str | None = None) -> list[dict[str, str]]:
    resolved = resolve_path(path)
    rows: list[dict[str, str]] = []
    with resolved.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if split and row["split"] != split:
                continue
            rows.append(row)
    return rows


class ImageClassificationDataset:
    def __init__(
        self,
        rows: list[dict[str, str]],
        class_to_idx: dict[str, int],
        transform: Callable | None = None,
    ) -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow no esta instalado. Ejecuta `uv sync`.") from exc

        self._image_cls = Image
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = self._image_cls.open(resolve_path(row["path"])).convert("RGB")
        label = self.class_to_idx[row["class_name"]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label, row


def collate_with_rows(batch):
    try:
        from torch.utils.data import default_collate
    except ImportError as exc:
        raise RuntimeError("torch no esta instalado. Ejecuta `uv sync`.") from exc

    images, labels, rows = zip(*batch)
    return default_collate(images), default_collate(labels), list(rows)
