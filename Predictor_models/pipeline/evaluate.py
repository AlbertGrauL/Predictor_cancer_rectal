from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from .config import load_config
from .dataset import ImageClassificationDataset, collate_with_rows, load_manifest
from .metrics import compute_classification_metrics, save_curves
from .models import build_model
from .transforms import build_transforms
from .utils import dependency_guard, load_paths, resolve_path, to_project_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evalua un checkpoint entrenado.")
    parser.add_argument("--config", default="Predictor_models/configs/multiclass_baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default="Predictor_models/artifacts/manifests/dataset_manifest.csv")
    parser.add_argument("--max-samples", type=int, default=None, help="Limita muestras de test para smoke tests.")
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
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[list[float]] = []
    details: list[dict[str, str | float | int]] = []
    with torch.no_grad():
        for images, labels, rows in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            label_list = labels.cpu().tolist()
            pred_list = predictions.cpu().tolist()
            score_list = probabilities.cpu().tolist()
            y_true.extend(label_list)
            y_pred.extend(pred_list)
            y_score.extend(score_list)
            for row, truth, pred, score in zip(rows, label_list, pred_list, score_list):
                details.append(
                    {
                        "path": row["path"],
                        "source_name": row["source_name"],
                        "class_name": row["class_name"],
                        "y_true": truth,
                        "y_pred": pred,
                        "score_polipo": score[positive_index] if positive_index < len(score) else None,
                        "probabilities": score,
                    }
                )
    return y_true, y_pred, y_score, details


def external_eval(config: dict, class_to_idx: dict[str, int], model, device, threshold: float, positive_index: int) -> dict:
    import torch
    from torch.utils.data import DataLoader

    paths = load_paths(config)
    dataset_cfg = config["dataset"]
    extensions = {ext.lower() for ext in dataset_cfg["extensions"]}
    _, eval_transform = build_transforms(dataset_cfg["image_size"])

    if not dataset_cfg.get("external_eval_sources"):
        return {}

    summary: dict[str, dict[str, float | int]] = {}
    for group_name, relative_paths in dataset_cfg.get("external_eval_sources", {}).items():
        rows = []
        for relative_path in relative_paths:
            for file_path in (paths.dataset_root / relative_path).rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    rows.append(
                        {
                            "path": to_project_relative(file_path),
                            "class_name": next(iter(class_to_idx.keys())),
                            "source_name": relative_path,
                            "file_name": file_path.name,
                            "split": "external",
                        }
                    )
        if not rows:
            summary[group_name] = {"images": 0, "predicted_as_polipo": 0, "rate": 0.0}
            continue

        dataset = ImageClassificationDataset(rows, class_to_idx, transform=eval_transform)
        loader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_with_rows,
        )
        predicted_as_polipo = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, _labels, _rows in loader:
                images = images.to(device)
                probabilities = torch.softmax(model(images), dim=1)[:, positive_index]
                predicted_as_polipo += int((probabilities >= threshold).sum().item())
                total += images.size(0)
        summary[group_name] = {
            "images": total,
            "predicted_as_polipo": predicted_as_polipo,
            "rate": predicted_as_polipo / total if total else 0.0,
        }
    return summary


def main() -> None:
    dependency_guard(
        {
            "torch": "torch",
            "torchvision": "torchvision",
            "numpy": "numpy",
            "yaml": "pyyaml",
            "PIL": "pillow",
            "sklearn": "scikit-learn",
            "matplotlib": "matplotlib",
        }
    )
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config)
    class_names = list(config["dataset"]["classes"].keys())
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    positive_index = class_to_idx[config["training"]["positive_class_name"]]

    checkpoint = torch.load(resolve_path(args.checkpoint), map_location="cpu")
    model_name = checkpoint["model_name"]
    model = build_model(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    _, eval_transform = build_transforms(config["dataset"]["image_size"])
    test_rows = load_manifest(args.manifest, split="test")
    if args.max_samples:
        test_rows = sample_rows(test_rows, args.max_samples, config["project"]["random_seed"])
    test_dataset = ImageClassificationDataset(test_rows, class_to_idx, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        collate_fn=collate_with_rows,
    )

    y_true, y_pred, y_score, details = evaluate_loader(
        model,
        test_loader,
        device=device,
        threshold=config["training"].get("threshold", 0.5),
        positive_index=positive_index,
    )
    metrics = compute_classification_metrics(y_true, y_pred, y_score, class_names, positive_index)
    curve_paths = save_curves(y_true, y_score, paths.figures_dir, prefix=model_name, class_names=class_names)

    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "errors": 0})
    for item in details:
        source = str(item["source_name"])
        by_source[source]["total"] += 1
        if item["y_true"] != item["y_pred"]:
            by_source[source]["errors"] += 1

    report = {
        "model_name": model_name,
        "checkpoint": to_project_relative(args.checkpoint),
        "class_names": class_names,
        "metrics": metrics,
        "curve_paths": curve_paths,
        "by_source": dict(by_source),
        "hard_cases": [item for item in details if item["y_true"] != item["y_pred"]][:25],
        "external_eval": external_eval(
            config,
            class_to_idx,
            model,
            device,
            config["training"].get("threshold", 0.5),
            positive_index,
        ),
    }

    report_path = paths.metrics_dir / f"{model_name}_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReporte guardado en: {to_project_relative(report_path)}")


if __name__ == "__main__":
    main()
