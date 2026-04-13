from __future__ import annotations


class BottomLeftMask:
    def __init__(self, width_ratio: float, height_ratio: float, fill: int = 0) -> None:
        self.width_ratio = width_ratio
        self.height_ratio = height_ratio
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        mask_width = max(1, int(width * self.width_ratio))
        mask_height = max(1, int(height * self.height_ratio))
        masked = image.copy()
        masked.paste(self.fill, (0, height - mask_height, mask_width, height))
        return masked


def build_transforms(image_size: int, preprocessing: dict | None = None):
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("torchvision no esta instalado. Ejecuta `uv sync`.") from exc

    preprocessing = preprocessing or {}
    mask_cfg = preprocessing.get("bottom_left_mask", {})
    base_steps = []
    if mask_cfg.get("enabled"):
        base_steps.append(
            BottomLeftMask(
                width_ratio=float(mask_cfg.get("width_ratio", 0.18)),
                height_ratio=float(mask_cfg.get("height_ratio", 0.18)),
                fill=int(mask_cfg.get("fill", 0)),
            )
        )

    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose(
        base_steps
        + [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            normalization,
        ]
    )
    eval_transform = transforms.Compose(
        base_steps
        + [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return train_transform, eval_transform
