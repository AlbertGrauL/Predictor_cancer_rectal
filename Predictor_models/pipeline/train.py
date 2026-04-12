from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from .config import load_config
from .dataset import ImageClassificationDataset, collate_with_rows, load_manifest
from .models import build_model, freeze_backbone, unfreeze_all
from .transforms import build_transforms
from .utils import dependency_guard, load_paths, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un modelo CNN para clasificacion binaria.")
    parser.add_argument("--config", default="Predictor_models/configs/binary_baseline.yaml")
    parser.add_argument("--model", default=None, help="Nombre del modelo a entrenar.")
    parser.add_argument("--manifest", default="Predictor_models/artifacts/manifests/dataset_manifest.csv")
    parser.add_argument("--epochs", type=int, default=None, help="Sobrescribe el numero de epocas.")
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Limita muestras para smoke tests.")
    return parser.parse_args()


def sample_rows(rows, limit: int | None, seed: int):
    if not limit or len(rows) <= limit:
        return rows
    sampled = list(rows)
    random.Random(seed).shuffle(sampled)
    return sampled[:limit]


def evaluate_loader(model, loader, device, threshold: float, positive_index: int):
    import torch

    model.eval()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    with torch.no_grad():
        for images, labels, _rows in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)[:, positive_index]
            predictions = (probabilities >= threshold).long()
            binary_labels = (labels == positive_index).long()

            total_loss += float(loss.item()) * images.size(0)
            y_true.extend(binary_labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            y_score.extend(probabilities.cpu().tolist())
    return total_loss / max(len(loader.dataset), 1), y_true, y_pred, y_score


def main() -> None:
    dependency_guard(
        {
            "torch": "torch",
            "torchvision": "torchvision",
            "numpy": "numpy",
            "yaml": "pyyaml",
            "PIL": "pillow",
            "sklearn": "scikit-learn",
        }
    )
    import torch
    from torch.utils.data import DataLoader

    from .metrics import compute_classification_metrics

    args = parse_args()
    config = load_config(args.config)
    set_seed(config["project"]["random_seed"])
    paths = load_paths(config)

    model_name = args.model or config["models"]["baseline"]
    manifest_path = Path(args.manifest)
    class_names = list(config["dataset"]["classes"].keys())
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    positive_index = class_to_idx[config["training"]["positive_class_name"]]
    image_size = config["dataset"]["image_size"]
    train_transform, eval_transform = build_transforms(image_size)

    train_rows = load_manifest(manifest_path, split="train")
    val_rows = load_manifest(manifest_path, split="val")
    if args.max_samples_per_split:
        train_rows = sample_rows(train_rows, args.max_samples_per_split, config["project"]["random_seed"])
        val_rows = sample_rows(val_rows, args.max_samples_per_split, config["project"]["random_seed"] + 1)

    train_dataset = ImageClassificationDataset(train_rows, class_to_idx, transform=train_transform)
    val_dataset = ImageClassificationDataset(val_rows, class_to_idx, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_with_rows,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_with_rows,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_name, num_classes=len(class_names), pretrained=True)
    model.to(device)
    freeze_backbone(model)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_path = paths.checkpoints_dir / f"{model_name}_best.pt"
    patience = 0
    threshold = config["training"]["threshold"]

    total_epochs = args.epochs or config["training"]["epochs"]
    for epoch in range(1, total_epochs + 1):
        if epoch == 2:
            unfreeze_all(model)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config["training"]["learning_rate"] / 10,
                weight_decay=config["training"]["weight_decay"],
            )

        model.train()
        running_loss = 0.0
        for images, labels, _rows in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * images.size(0)

        train_loss = running_loss / max(len(train_loader.dataset), 1)
        val_loss, y_true, y_pred, y_score = evaluate_loader(model, val_loader, device, threshold, positive_index)
        metrics = compute_classification_metrics(y_true, y_pred, y_score)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "class_names": class_names,
                    "config": config,
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= config["training"]["early_stopping_patience"]:
                break

    metadata_path = paths.metrics_dir / f"{model_name}_metadata.json"
    history_path = paths.metrics_dir / f"{model_name}_history.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "class_names": class_names,
                "checkpoint_path": str(best_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Checkpoint guardado en: {best_path}")
    print(f"Historial guardado en: {history_path}")
    print(f"Metadatos guardados en: {metadata_path}")


if __name__ == "__main__":
    main()
