from __future__ import annotations

import argparse
import random

from ..config import load_config
from .dataset import load_manifest
from .transforms import BottomLeftMask, build_transforms
from ..utils import ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera una vista previa del preprocesado aplicado a una imagen.")
    parser.add_argument("--config", default="Predictor_models/configs/image/multiclass_baseline.yaml")
    parser.add_argument("--image", required=True, help="Ruta de la imagen de entrada.")
    parser.add_argument("--manifest", default="Predictor_models/artifacts/manifests/dataset_manifest.csv")
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--output",
        default="Predictor_models/artifacts/reports/preprocessing_preview.png",
        help="Ruta de salida para la imagen preprocesada.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow no esta instalado. Ejecuta `uv sync`.") from exc

    args = parse_args()
    config = load_config(args.config)
    mask_cfg = config.get("preprocessing", {}).get("bottom_left_mask", {})
    augmentation = config.get("augmentation", {})

    image_path = resolve_path(args.image)
    output_path = resolve_path(args.output)
    ensure_dir(output_path.parent)

    image = Image.open(image_path).convert("RGB")
    if mask_cfg.get("enabled"):
        image = BottomLeftMask(
            width_ratio=float(mask_cfg.get("width_ratio", 0.30)),
            height_ratio=float(mask_cfg.get("height_ratio", 0.35)),
            fill=int(mask_cfg.get("fill", 0)),
        )(image)
    image.save(output_path)

    print(f"Vista previa guardada en: {output_path}")

    rows = load_manifest(args.manifest, split=args.split)
    if not rows:
        return

    preview_count = max(1, min(int(args.count), 8))
    random.Random(config["project"]["random_seed"]).shuffle(rows)
    selected_rows = rows[:preview_count]
    train_transform, _eval_transform = build_transforms(
        config["dataset"]["image_size"],
        preprocessing=config.get("preprocessing", {}),
        augmentation=augmentation,
    )

    preview_dir = ensure_dir(output_path.parent / "augmentation_preview_samples")
    for index, row in enumerate(selected_rows, start=1):
        sample_image = Image.open(resolve_path(row["path"])).convert("RGB")
        transformed = train_transform(sample_image)
        restored = transformed.detach().cpu().permute(1, 2, 0).numpy()
        restored = restored * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        restored = (restored.clip(0, 1) * 255).astype("uint8")
        sample_output = preview_dir / f"sample_{index}_{row['class_name']}.png"
        Image.fromarray(restored).save(sample_output)
    print(f"Muestras de augmentacion guardadas en: {preview_dir}")
