from __future__ import annotations

import argparse
from collections import Counter
import json
import random

from ..config import load_config
from .dataset import ImageClassificationDataset, collate_with_rows, load_manifest
from .models import build_model, freeze_backbone, unfreeze_all
from .transforms import build_transforms
from ..utils import dependency_guard, load_paths, resolve_path, set_seed, to_project_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena un modelo CNN para clasificacion multiclase.")
    parser.add_argument("--config", default="Predictor_models/configs/image/multiclass_baseline.yaml")
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


def evaluate_loader(model, loader, device, positive_index: int):
    import torch

    model.eval()
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[list[float]] = []
    with torch.no_grad():
        for images, labels, _rows in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            total_loss += float(loss.item()) * images.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            y_score.extend(probabilities.cpu().tolist())
    return total_loss / max(len(loader.dataset), 1), y_true, y_pred, y_score


def build_optimizer(model, optimizer_cfg: dict):
    import torch

    optimizer_name = str(optimizer_cfg.get("name", "adamw")).lower()
    learning_rate = float(optimizer_cfg.get("learning_rate", 0.0003))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0001))

    if optimizer_name != "adamw":
        raise ValueError(f"Optimizador no soportado: {optimizer_name}")

    return torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def build_scheduler(optimizer, scheduler_cfg: dict):
    import torch

    scheduler_name = str(scheduler_cfg.get("name", "none")).lower()
    if scheduler_name in {"none", ""}:
        return None
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(scheduler_cfg.get("mode", "max")),
            factor=float(scheduler_cfg.get("factor", 0.5)),
            patience=int(scheduler_cfg.get("patience", 2)),
        )
    raise ValueError(f"Scheduler no soportado: {scheduler_name}")


def compute_class_weights(train_rows, class_to_idx: dict[str, int]):
    import torch

    counts = Counter(row["class_name"] for row in train_rows)
    total = sum(counts.values())
    class_count = len(class_to_idx)
    weights = []
    for class_name, _class_index in sorted(class_to_idx.items(), key=lambda item: item[1]):
        count = counts.get(class_name, 1)
        weights.append(total / max(count * class_count, 1))
    return torch.tensor(weights, dtype=torch.float32)


def build_sampler(train_rows, enabled: bool):
    if not enabled:
        return None
    import torch
    from torch.utils.data import WeightedRandomSampler

    counts = Counter(row["class_name"] for row in train_rows)
    sample_weights = [1.0 / counts[row["class_name"]] for row in train_rows]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_rows),
        replacement=True,
    )


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

    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})
    loss_cfg = config.get("loss", {})
    sampling_cfg = config.get("sampling", {})
    preprocessing = config.get("preprocessing", {})
    augmentation = config.get("augmentation", {})

    model_name = args.model or config["models"]["baseline"]
    manifest_path = resolve_path(args.manifest)
    class_names = list(config["dataset"]["classes"].keys())
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    positive_index = class_to_idx[config["training"]["positive_class_name"]]
    image_size = config["dataset"]["image_size"]
    train_transform, eval_transform = build_transforms(
        image_size,
        preprocessing=preprocessing,
        augmentation=augmentation,
    )

    train_rows = load_manifest(manifest_path, split="train")
    val_rows = load_manifest(manifest_path, split="val")
    if args.max_samples_per_split:
        train_rows = sample_rows(train_rows, args.max_samples_per_split, config["project"]["random_seed"])
        val_rows = sample_rows(val_rows, args.max_samples_per_split, config["project"]["random_seed"] + 1)

    train_dataset = ImageClassificationDataset(train_rows, class_to_idx, transform=train_transform)
    val_dataset = ImageClassificationDataset(val_rows, class_to_idx, transform=eval_transform)

    sampler = build_sampler(train_rows, enabled=bool(sampling_cfg.get("use_weighted_sampler", False)))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
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

    class_weights = None
    if loss_cfg.get("use_class_weights", False):
        class_weights = compute_class_weights(train_rows, class_to_idx).to(device)

    loss_name = str(loss_cfg.get("name", "cross_entropy")).lower()
    if loss_name != "cross_entropy":
        raise ValueError(f"Loss no soportada: {loss_name}")
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = build_optimizer(model, optimizer_cfg)
    scheduler = build_scheduler(optimizer, scheduler_cfg)

    history: list[dict[str, float | int | None]] = []
    best_f1 = -1.0
    best_path = paths.checkpoints_dir / f"{model_name}_best.pt"
    patience = 0

    training_setup = {
        "optimizer": optimizer_cfg,
        "scheduler": scheduler_cfg,
        "loss": loss_cfg,
        "sampling": sampling_cfg,
        "augmentation": augmentation,
        "preprocessing": preprocessing,
    }

    total_epochs = args.epochs or config["training"]["epochs"]
    for epoch in range(1, total_epochs + 1):
        if epoch == 2:
            unfreeze_all(model)
            fine_tune_optimizer_cfg = dict(optimizer_cfg)
            fine_tune_optimizer_cfg["learning_rate"] = float(optimizer_cfg.get("learning_rate", 0.0003)) / 10.0
            optimizer = build_optimizer(model, fine_tune_optimizer_cfg)
            scheduler = build_scheduler(optimizer, scheduler_cfg)

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
        val_loss, y_true, y_pred, y_score = evaluate_loader(model, val_loader, device, positive_index)
        metrics = compute_classification_metrics(y_true, y_pred, y_score, class_names, positive_index)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_precision_macro": metrics["precision_macro"],
            "val_recall_macro": metrics["recall_macro"],
            "val_f1_macro": metrics["f1_macro"],
            "val_f1_weighted": metrics["f1_weighted"],
            "val_roc_auc": metrics["roc_auc"],
            "val_pr_auc": metrics["pr_auc"],
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False))

        score_metric_name = config["training"].get("primary_metric", "f1_macro")
        score_metric_value = metrics.get(score_metric_name) or metrics.get("f1_macro") or metrics.get("f1")
        if scheduler is not None:
            scheduler.step(score_metric_value)

        if score_metric_value > best_f1:
            best_f1 = score_metric_value
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "class_names": class_names,
                    "config": config,
                    "primary_metric": score_metric_name,
                    "training_setup": training_setup,
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
                "modality": "image",
                "class_names": class_names,
                "checkpoint_path": to_project_relative(best_path),
                "training_setup": training_setup,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Checkpoint guardado en: {to_project_relative(best_path)}")
    print(f"Historial guardado en: {to_project_relative(history_path)}")
    print(f"Metadatos guardados en: {to_project_relative(metadata_path)}")


if __name__ == "__main__":
    main()
