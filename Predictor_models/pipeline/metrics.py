from __future__ import annotations

from pathlib import Path

from .utils import to_project_relative


def compute_classification_metrics(y_true, y_pred, y_score, class_names: list[str], positive_index: int | None = None):
    try:
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.preprocessing import label_binarize
    except ImportError as exc:
        raise RuntimeError("scikit-learn/numpy no estan instalados. Ejecuta `uv sync`.") from exc

    y_score_arr = np.asarray(y_score)
    if y_score_arr.ndim == 1:
        y_score_arr = np.column_stack([1 - y_score_arr, y_score_arr])

    n_classes = len(class_names)
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    precision_per_class = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    metrics = {
        "metric_scope": "binary" if n_classes == 2 else "multiclass",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "per_class": {
            class_name: {
                "precision": float(precision_per_class[index]),
                "recall": float(recall_per_class[index]),
                "f1": float(f1_per_class[index]),
                "support": int(cm[index].sum()),
            }
            for index, class_name in enumerate(class_names)
        },
    }

    if n_classes == 2 and positive_index is not None:
        y_true_pos = [1 if value == positive_index else 0 for value in y_true]
        y_pred_pos = [1 if value == positive_index else 0 for value in y_pred]
        positive_scores = y_score_arr[:, positive_index]
        metrics["precision_positive"] = float(precision_score(y_true_pos, y_pred_pos, zero_division=0))
        metrics["recall_positive"] = float(recall_score(y_true_pos, y_pred_pos, zero_division=0))
        metrics["f1_positive"] = float(f1_score(y_true_pos, y_pred_pos, zero_division=0))
        metrics["roc_auc"] = float(roc_auc_score(y_true_pos, positive_scores)) if len(set(y_true_pos)) > 1 else None
        metrics["pr_auc"] = (
            float(average_precision_score(y_true_pos, positive_scores)) if len(set(y_true_pos)) > 1 else None
        )
    else:
        y_true_bin = label_binarize(y_true, classes=labels)
        metrics["roc_auc"] = (
            float(roc_auc_score(y_true_bin, y_score_arr, multi_class="ovr", average="macro"))
            if len(set(y_true)) > 1
            else None
        )
        metrics["pr_auc"] = (
            float(average_precision_score(y_true_bin, y_score_arr, average="macro"))
            if len(set(y_true)) > 1
            else None
        )

    return metrics


def save_curves(y_true, y_score, figures_dir: Path, prefix: str, class_names: list[str]) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
        from sklearn.preprocessing import label_binarize
    except ImportError as exc:
        raise RuntimeError("matplotlib/scikit-learn/seaborn/numpy no estan instalados. Ejecuta `uv sync`.") from exc

    figures_dir.mkdir(parents=True, exist_ok=True)
    score_arr = np.asarray(y_score)
    if score_arr.ndim == 1:
        score_arr = np.column_stack([1 - score_arr, score_arr])

    labels = list(range(len(class_names)))
    y_true_bin = label_binarize(y_true, classes=labels)
    if len(class_names) == 2 and y_true_bin.ndim == 2 and y_true_bin.shape[1] == 1:
        y_true_bin = np.column_stack([1 - y_true_bin[:, 0], y_true_bin[:, 0]])
    outputs: dict[str, str] = {}

    roc_path = figures_dir / f"{prefix}_roc_curve.png"
    plt.figure(figsize=(6, 5))
    for index, class_name in enumerate(class_names):
        if y_true_bin[:, index].sum() == 0:
            continue
        RocCurveDisplay.from_predictions(y_true_bin[:, index], score_arr[:, index], name=class_name)
    plt.title("Curvas ROC por clase")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()
    outputs["roc_curve"] = to_project_relative(roc_path)

    pr_path = figures_dir / f"{prefix}_pr_curve.png"
    plt.figure(figsize=(6, 5))
    for index, class_name in enumerate(class_names):
        if y_true_bin[:, index].sum() == 0:
            continue
        PrecisionRecallDisplay.from_predictions(y_true_bin[:, index], score_arr[:, index], name=class_name)
    plt.title("Curvas Precision-Recall por clase")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()
    outputs["pr_curve"] = to_project_relative(pr_path)

    cm_path = figures_dir / f"{prefix}_confusion_matrix.png"
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, np.argmax(score_arr, axis=1), labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de confusión")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150)
    plt.close()
    outputs["confusion_matrix_plot"] = to_project_relative(cm_path)

    if len(class_names) == 2:
        positive_scores = score_arr[:, 1]
        positive_truth = y_true_bin[:, 1]
        prob_true, prob_pred = calibration_curve(positive_truth, positive_scores, n_bins=10)
        calibration_path = figures_dir / f"{prefix}_calibration_curve.png"
        plt.figure(figsize=(6, 4))
        plt.plot(prob_pred, prob_true, marker="o", label="Modelo")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
        plt.xlabel("Probabilidad media predicha")
        plt.ylabel("Frecuencia observada")
        plt.title("Curva de calibración")
        plt.legend()
        plt.tight_layout()
        plt.savefig(calibration_path, dpi=150)
        plt.close()
        outputs["calibration_curve"] = to_project_relative(calibration_path)

    return outputs
