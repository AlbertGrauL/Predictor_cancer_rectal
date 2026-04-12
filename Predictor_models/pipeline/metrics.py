from __future__ import annotations

from pathlib import Path


def compute_classification_metrics(y_true, y_pred, y_score):
    try:
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError as exc:
        raise RuntimeError("scikit-learn no esta instalado. Ejecuta `uv sync`.") from exc

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def save_curves(y_true, y_score, figures_dir: Path, prefix: str) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
    except ImportError as exc:
        raise RuntimeError("matplotlib/scikit-learn no estan instalados. Ejecuta `uv sync`.") from exc

    figures_dir.mkdir(parents=True, exist_ok=True)

    roc_path = figures_dir / f"{prefix}_roc_curve.png"
    RocCurveDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    pr_path = figures_dir / f"{prefix}_pr_curve.png"
    PrecisionRecallDisplay.from_predictions(y_true, y_score)
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10)
    calibration_path = figures_dir / f"{prefix}_calibration_curve.png"
    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker="o", label="Modelo")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Ideal")
    plt.xlabel("Probabilidad media predicha")
    plt.ylabel("Frecuencia observada")
    plt.title("Curva de calibracion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(calibration_path, dpi=150)
    plt.close()

    return {
        "roc_curve": str(roc_path),
        "pr_curve": str(pr_path),
        "calibration_curve": str(calibration_path),
    }
