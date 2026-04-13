from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .transforms import BottomLeftMask
from .utils import ensure_dir, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera una vista previa del preprocesado aplicado a una imagen.")
    parser.add_argument("--config", default="Predictor_models/configs/multiclass_baseline.yaml")
    parser.add_argument("--image", required=True, help="Ruta de la imagen de entrada.")
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


if __name__ == "__main__":
    main()
