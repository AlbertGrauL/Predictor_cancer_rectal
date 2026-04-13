from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
else:
    ROOT = Path(__file__).resolve().parents[1]

from Predictor_models.pipeline.config import load_config
from Predictor_models.pipeline.inference import Predictor
from Predictor_models.pipeline.utils import resolve_path


ARTIFACTS_ROOT = ROOT / "Predictor_models" / "artifacts"
METRICS_DIR = ARTIFACTS_ROOT / "metrics"
FIGURES_DIR = ARTIFACTS_ROOT / "figures"
ACTIVE_CONFIG_PATH = ROOT / "Predictor_models" / "configs" / "multiclass_baseline.yaml"
ACTIVE_CONFIG = load_config(str(ACTIVE_CONFIG_PATH))
EXPECTED_CLASS_COUNT = len(ACTIVE_CONFIG["dataset"]["classes"])
DEFAULT_MODEL_NAME = "efficientnet_b0"
METRIC_HELP = {
    "Accuracy": "Porcentaje total de predicciones correctas sobre todas las imágenes.",
    "Recall macro": "Promedio del recall de todas las clases. Indica si el modelo detecta bien cada clase sin favorecer solo a una.",
    "F1 macro": "Promedio equilibrado entre precision y recall en todas las clases. Es una métrica muy útil en clasificación multiclase.",
    "F1 weighted": "Versión ponderada del F1 según el tamaño de cada clase. Da más peso a las clases con más imágenes.",
    "ROC-AUC": "Mide la capacidad del modelo para separar clases usando un esquema uno contra resto. Más alto suele indicar mejor discriminación.",
    "PR-AUC": "Resume la relación precision-recall también en esquema uno contra resto. Es útil cuando una clase es más crítica.",
}
SECTION_HELP = {
    "comparison_table": "Esta tabla resume el rendimiento guardado de cada modelo. Sirve para comparar rápidamente qué arquitectura ha funcionado mejor.",
    "comparison_chart": "Este gráfico compara visualmente varias métricas clave entre modelos. Permite ver de un vistazo qué red está más equilibrada.",
    "detail_metrics": "Estas métricas pertenecen solo al modelo seleccionado y resumen su comportamiento global en el conjunto de evaluación.",
    "roc_curve": "La curva ROC muestra cómo cambia la detección al variar el umbral de decisión. Cuanto más se acerca a la esquina superior izquierda, mejor.",
    "pr_curve": "La curva Precision-Recall muestra el equilibrio entre detectar más pólipos y cometer menos falsos positivos.",
    "calibration_curve": "La curva de calibración compara la confianza del modelo con lo que realmente ocurre. Sirve para ver si sus probabilidades son fiables.",
    "external_eval": "La evaluación externa mide cómo responde el modelo ante patologías que no formaron parte del entrenamiento principal.",
    "hard_cases": "Estos casos son ejemplos donde el modelo se equivoca. Ayudan a entender sus debilidades y limitaciones.",
    "prediction_probs": "Este gráfico muestra la probabilidad asignada a cada clase para la imagen cargada.",
    "gradcam": "Grad-CAM resalta las zonas de la imagen que más han influido en la decisión del modelo.",
}


def load_image(image_file) -> Image.Image:
    return Image.open(image_file).convert("RGB")


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    import matplotlib.cm as cm

    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(image.size)
    heatmap_array = np.asarray(heatmap_resized, dtype=np.float32) / 255.0
    colored = cm.jet(heatmap_array)[..., :3]
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    blended = 0.55 * image_array + 0.45 * colored
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def explanation_text(predicted_class: str, confidence: float) -> str:
    label = predicted_class.replace("_", " ")
    return (
        f"El modelo estima que la imagen pertenece a la clase **{label}** "
        f"con una confianza aproximada del **{confidence:.1%}**. "
        "La visualizacion Grad-CAM resalta las regiones que mas han influido en la decision."
    )


def model_label(model_name: str) -> str:
    return model_name.replace("_", " ").upper()


def safe_json(path: Path) -> dict | None:
    resolved = resolve_path(path)
    if not resolved.exists():
        return None
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return None


def discover_models() -> list[dict]:
    models: list[dict] = []
    for metadata_path in sorted(METRICS_DIR.glob("*_metadata.json")):
        metadata = safe_json(metadata_path)
        if not metadata:
            continue
        class_names = metadata.get("class_names", [])
        if len(class_names) != EXPECTED_CLASS_COUNT:
            continue
        model_name = metadata["model_name"]
        evaluation_path = METRICS_DIR / f"{model_name}_evaluation.json"
        history_path = METRICS_DIR / f"{model_name}_history.json"
        checkpoint_path = resolve_path(metadata["checkpoint_path"])
        evaluation = safe_json(evaluation_path)
        history = safe_json(history_path)
        models.append(
            {
                "model_name": model_name,
                "label": model_label(model_name),
                "metadata_path": metadata_path,
                "checkpoint_path": checkpoint_path,
                "class_names": class_names,
                "evaluation": evaluation,
                "history": history,
                "available_for_prediction": checkpoint_path.exists(),
            }
        )
    return models


def comparison_dataframe(models: list[dict]) -> pd.DataFrame:
    rows = []
    for model in models:
        evaluation = model["evaluation"] or {}
        metrics = evaluation.get("metrics", {})
        rows.append(
            {
                "Modelo": model["label"],
                "Archivo": model["model_name"],
                "Clases": len(model.get("class_names", [])),
                "Disponible": "Sí" if model["available_for_prediction"] else "No",
                "Accuracy": metrics.get("accuracy"),
                "Recall macro": metrics.get("recall_macro", metrics.get("recall")),
                "F1 macro": metrics.get("f1_macro", metrics.get("f1")),
                "F1 weighted": metrics.get("f1_weighted", metrics.get("f1")),
                "ROC-AUC": metrics.get("roc_auc"),
                "PR-AUC": metrics.get("pr_auc"),
            }
        )
    dataframe = pd.DataFrame(rows)
    if not dataframe.empty:
        dataframe = dataframe.sort_values(by=["F1 macro", "Recall macro", "Accuracy"], ascending=False, na_position="last")
    return dataframe


def metric_value(metrics: dict, *keys: str):
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def format_metric(value) -> str:
    return f"{value:.3f}" if value is not None else "N/D"


@st.cache_resource(show_spinner=False)
def load_predictor_cached(checkpoint_path: str, metadata_path: str) -> Predictor:
    return Predictor(checkpoint_path=checkpoint_path, metadata_path=metadata_path)


def render_model_overview(selected_model: dict, models: list[dict]) -> None:
    st.subheader("Comparativa de modelos")
    comparison_df = comparison_dataframe(models)
    if comparison_df.empty:
        st.warning("Todavía no hay modelos con metadatos registrados.")
        return

    with st.expander("Tabla comparativa de modelos", expanded=True):
        st.write(SECTION_HELP["comparison_table"])
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    with st.expander("Qué significa cada métrica", expanded=False):
        for metric_name, description in METRIC_HELP.items():
            st.markdown(f"**{metric_name}**: {description}")

    with st.expander("Gráfico comparativo entre modelos", expanded=True):
        st.write(SECTION_HELP["comparison_chart"])
        chart_df = comparison_df[["Modelo", "F1 macro", "Recall macro", "Accuracy"]].set_index("Modelo")
        st.bar_chart(chart_df, use_container_width=True)

    selected_eval = selected_model["evaluation"]
    if not selected_eval:
        st.info("El modelo seleccionado no tiene evaluación guardada todavía.")
        return

    metrics = selected_eval["metrics"]
    st.subheader(f"Detalle de {selected_model['label']}")
    with st.expander("Métricas del modelo seleccionado", expanded=True):
        st.write(SECTION_HELP["detail_metrics"])
        metrics = selected_eval["metrics"]
        top_metrics = st.columns(5)
        top_metrics[0].metric("Accuracy", format_metric(metric_value(metrics, "accuracy")))
        top_metrics[1].metric("Recall macro", format_metric(metric_value(metrics, "recall_macro", "recall")))
        top_metrics[2].metric("F1 macro", format_metric(metric_value(metrics, "f1_macro", "f1")))
        top_metrics[3].metric("ROC-AUC", format_metric(metric_value(metrics, "roc_auc")))
        top_metrics[4].metric("PR-AUC", format_metric(metric_value(metrics, "pr_auc")))

        for metric_name, description in METRIC_HELP.items():
            st.caption(f"{metric_name}: {description}")

    per_class = selected_eval["metrics"].get("per_class", {})
    if per_class:
        with st.expander("Métricas por clase", expanded=False):
            st.write("Esta tabla muestra cómo rinde el modelo en cada clase individual.")
            per_class_df = pd.DataFrame(
                [
                    {
                        "Clase": class_name,
                        "Precision": values.get("precision"),
                        "Recall": values.get("recall"),
                        "F1": values.get("f1"),
                        "Soporte": values.get("support"),
                    }
                    for class_name, values in per_class.items()
                ]
            )
            st.dataframe(per_class_df, use_container_width=True, hide_index=True)
            st.bar_chart(per_class_df.set_index("Clase")[["Recall", "F1"]], use_container_width=True)

    curves = selected_eval.get("curve_paths", {})
    with st.expander("Curva ROC", expanded=False):
        info_col, image_col = st.columns([1.1, 1])
        info_col.write(SECTION_HELP["roc_curve"])
        curve_path = curves.get("roc_curve")
        resolved_curve = resolve_path(curve_path) if curve_path else None
        if resolved_curve and resolved_curve.exists():
            image_col.image(str(resolved_curve), caption="Curva ROC", width=320)
        else:
            image_col.info("Curva ROC no disponible")

    with st.expander("Curva Precision-Recall", expanded=False):
        info_col, image_col = st.columns([1.1, 1])
        info_col.write(SECTION_HELP["pr_curve"])
        curve_path = curves.get("pr_curve")
        resolved_curve = resolve_path(curve_path) if curve_path else None
        if resolved_curve and resolved_curve.exists():
            image_col.image(str(resolved_curve), caption="Curva Precision-Recall", width=320)
        else:
            image_col.info("Curva Precision-Recall no disponible")

    with st.expander("Curva de calibración", expanded=False):
        info_col, image_col = st.columns([1.1, 1])
        info_col.write(SECTION_HELP["calibration_curve"])
        curve_path = curves.get("calibration_curve")
        resolved_curve = resolve_path(curve_path) if curve_path else None
        if resolved_curve and resolved_curve.exists():
            image_col.image(str(resolved_curve), caption="Curva de calibración", width=320)
        else:
            image_col.info("Curva de calibración no disponible")

    with st.expander("Matriz de confusión", expanded=False):
        st.write("La matriz de confusión muestra cuántas imágenes de cada clase se clasifican correctamente y dónde se producen los errores.")
        curve_path = curves.get("confusion_matrix_plot")
        resolved_curve = resolve_path(curve_path) if curve_path else None
        if resolved_curve and resolved_curve.exists():
            st.image(str(resolved_curve), caption="Matriz de confusión", width=420)
        else:
            st.info("Matriz de confusión no disponible")

    external_eval = selected_eval.get("external_eval", {})
    if external_eval:
        with st.expander("Evaluación externa exploratoria", expanded=False):
            st.write(SECTION_HELP["external_eval"])
            st.write(external_eval)

    hard_cases = selected_eval.get("hard_cases", [])
    if hard_cases:
        with st.expander("Ejemplos de errores del modelo", expanded=False):
            st.write(SECTION_HELP["hard_cases"])
            st.dataframe(pd.DataFrame(hard_cases[:8]), use_container_width=True, hide_index=True)


def render_prediction_area(selected_model: dict) -> None:
    st.subheader("Predicción sobre imagen")
    with st.expander("Modelo activo para predicción", expanded=True):
        st.write(f"Estás utilizando **{selected_model['label']}** para generar la predicción de la imagen.")

    if not selected_model["available_for_prediction"]:
        st.error("El checkpoint del modelo seleccionado no está disponible para inferencia.")
        return

    uploaded_image = st.file_uploader("Sube una imagen endoscópica", type=["jpg", "jpeg", "png"])
    if uploaded_image is None:
        st.info("Sube una imagen para generar una predicción y su explicación visual.")
        return

    image = load_image(uploaded_image)

    try:
        predictor = load_predictor_cached(
            checkpoint_path=str(selected_model["checkpoint_path"]),
            metadata_path=str(selected_model["metadata_path"]),
        )
        temp_path = ARTIFACTS_ROOT / "tmp_streamlit_image.png"
        image.save(temp_path)
        result = predictor.predict(temp_path)
        overlay = overlay_heatmap(image, result["heatmap"])

        with st.expander("Resultado de la predicción", expanded=True):
            predicted_label = result["predicted_class"].replace("_", " ").upper()
            summary_cols = st.columns([1, 1, 1])
            summary_cols[0].metric("Clase predicha", predicted_label)
            summary_cols[1].metric("Confianza", f"{result['confidence']:.1%}")
            top_other = max(
                (value for key, value in result["probabilities"].items() if key != result["predicted_class"]),
                default=0.0,
            )
            summary_cols[2].metric("Diferencia frente a la siguiente", f"{(result['confidence'] - top_other):.1%}")
            st.success(explanation_text(result["predicted_class"], result["confidence"]))
            st.info(
                "La clase predicha es la que recibe mayor probabilidad. "
                "La diferencia frente a la siguiente ayuda a ver si la decisión es clara o si el modelo está dudando."
            )

        probabilities = pd.DataFrame(
            {
                "Clase": list(result["probabilities"].keys()),
                "Probabilidad": list(result["probabilities"].values()),
            }
        ).sort_values(by="Probabilidad", ascending=False)

        with st.expander("Mapa Grad-CAM", expanded=True):
            st.write(SECTION_HELP["gradcam"])
            image_cols = st.columns([1, 1])
            image_cols[0].image(image, caption="Imagen original", use_container_width=True)
            image_cols[1].image(overlay, caption="Zonas que más han influido en la decisión", use_container_width=True)

        with st.expander("Probabilidades por clase", expanded=True):
            st.write(SECTION_HELP["prediction_probs"])
            st.bar_chart(probabilities.set_index("Clase"))
            st.dataframe(probabilities, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"No se pudo ejecutar la inferencia: {exc}")


def main() -> None:
    st.set_page_config(page_title="Predictor de Polipos", layout="wide")
    st.title("Predictor académico multiclase de pólipos")
    st.caption("Herramienta educativa multiclase. No es un sistema diagnóstico.")

    models = discover_models()
    if not models:
        st.error("No se han encontrado modelos multiclase en Predictor_models/artifacts/metrics.")
        return

    default_index = 0
    for index, model in enumerate(models):
        if model["model_name"] == DEFAULT_MODEL_NAME:
            default_index = index
            break

    selected_label = st.selectbox(
        "Selecciona el modelo que quieres utilizar",
        options=[model["label"] for model in models],
        index=default_index,
    )
    selected_model = next(model for model in models if model["label"] == selected_label)

    overview_tab, prediction_tab = st.tabs(["Comparativa", "Predicción"])
    with overview_tab:
        render_model_overview(selected_model, models)
    with prediction_tab:
        render_prediction_area(selected_model)


if __name__ == "__main__":
    main()
