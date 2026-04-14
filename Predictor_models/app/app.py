from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
else:
    ROOT = Path(__file__).resolve().parents[2]

from Predictor_models.pipeline.config import load_config
from Predictor_models.pipeline.image.inference import Predictor
from Predictor_models.pipeline.tabular.tabular_inference import TabularPredictor
from Predictor_models.pipeline.tabular.tabular_utils import build_question_specs, load_questions
from Predictor_models.pipeline.utils import resolve_path


IMAGE_ARTIFACTS_ROOT = ROOT / "Predictor_models" / "artifacts"
IMAGE_METRICS_DIR = IMAGE_ARTIFACTS_ROOT / "metrics"
TABULAR_ARTIFACTS_ROOT = ROOT / "Predictor_models" / "artifacts" / "tabular"
TABULAR_METRICS_DIR = TABULAR_ARTIFACTS_ROOT / "metrics"
IMAGE_CONFIG_PATH = ROOT / "Predictor_models" / "configs" / "image" / "multiclass_baseline.yaml"
TABULAR_CONFIG_PATH = ROOT / "Predictor_models" / "configs" / "tabular" / "tabular_baseline.yaml"
IMAGE_CONFIG = load_config(str(IMAGE_CONFIG_PATH))
TABULAR_CONFIG = load_config(str(TABULAR_CONFIG_PATH))
DEFAULT_IMAGE_MODEL = "efficientnet_b0"
DEFAULT_TABULAR_MODEL = "xgboost"

IMAGE_METRIC_HELP = {
    "Accuracy": "Porcentaje global de imágenes clasificadas correctamente.",
    "Recall macro": "Promedio del recall de todas las clases de imagen.",
    "F1 macro": "Equilibrio entre precision y recall en las tres clases de imagen.",
    "ROC-AUC": "Capacidad general del modelo para separar clases en esquema uno contra resto.",
    "PR-AUC": "Relación precision-recall resumida para la tarea multiclase.",
}
TABULAR_METRIC_HELP = {
    "Accuracy": "Porcentaje global de pacientes clasificados correctamente.",
    "Precision positiva": "De todos los pacientes marcados como de riesgo, cuántos lo eran realmente.",
    "Recall positivo": "De todos los pacientes con riesgo real, cuántos detecta el modelo tabular.",
    "F1 positivo": "Equilibrio entre precision positiva y recall positivo.",
    "ROC-AUC": "Capacidad del modelo tabular para separar perfiles de riesgo y no riesgo.",
    "PR-AUC": "Métrica útil cuando interesa especialmente la clase positiva de riesgo.",
}


def safe_json(path: Path) -> dict | None:
    resolved = resolve_path(path)
    if not resolved.exists():
        return None
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except Exception:
        return None


def model_label(model_name: str) -> str:
    return model_name.replace("_", " ").upper()


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


def discover_image_models() -> list[dict]:
    expected_class_count = len(IMAGE_CONFIG["dataset"]["classes"])
    models: list[dict] = []
    for metadata_path in sorted(IMAGE_METRICS_DIR.glob("*_metadata.json")):
        metadata = safe_json(metadata_path)
        if not metadata:
            continue
        if metadata.get("modality") not in {None, "image"}:
            continue
        class_names = metadata.get("class_names", [])
        if len(class_names) != expected_class_count:
            continue
        model_name = metadata["model_name"]
        evaluation = safe_json(IMAGE_METRICS_DIR / f"{model_name}_evaluation.json")
        checkpoint_path = resolve_path(metadata["checkpoint_path"])
        models.append(
            {
                "model_name": model_name,
                "label": model_label(model_name),
                "metadata_path": metadata_path,
                "checkpoint_path": checkpoint_path,
                "class_names": class_names,
                "evaluation": evaluation,
                "available_for_prediction": checkpoint_path.exists(),
            }
        )
    return models


def discover_tabular_models() -> list[dict]:
    models: list[dict] = []
    for metadata_path in sorted(TABULAR_METRICS_DIR.glob("*_metadata.json")):
        metadata = safe_json(metadata_path)
        if not metadata or metadata.get("modality") != "tabular":
            continue
        model_name = metadata["model_name"]
        evaluation = safe_json(TABULAR_METRICS_DIR / f"{model_name}_evaluation.json")
        checkpoint_path = resolve_path(metadata["checkpoint_path"])
        models.append(
            {
                "model_name": model_name,
                "label": model_label(model_name),
                "metadata_path": metadata_path,
                "checkpoint_path": checkpoint_path,
                "class_names": metadata.get("class_names", []),
                "evaluation": evaluation,
                "available_for_prediction": checkpoint_path.exists(),
            }
        )
    return models


def image_comparison_dataframe(models: list[dict]) -> pd.DataFrame:
    rows = []
    for model in models:
        evaluation = model.get("evaluation") or {}
        metrics = evaluation.get("metrics", {})
        rows.append(
            {
                "Modelo": model["label"],
                "Accuracy": metrics.get("accuracy"),
                "Recall macro": metrics.get("recall_macro", metrics.get("recall")),
                "F1 macro": metrics.get("f1_macro", metrics.get("f1")),
                "ROC-AUC": metrics.get("roc_auc"),
                "PR-AUC": metrics.get("pr_auc"),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(by=["F1 macro", "Recall macro", "Accuracy"], ascending=False, na_position="last")
    return frame


def tabular_comparison_dataframe(models: list[dict]) -> pd.DataFrame:
    rows = []
    for model in models:
        evaluation = model.get("evaluation") or {}
        metrics = evaluation.get("metrics", {})
        rows.append(
            {
                "Modelo": model["label"],
                "Accuracy": metrics.get("accuracy"),
                "Precision positiva": metrics.get("precision_positive"),
                "Recall positivo": metrics.get("recall_positive"),
                "F1 positivo": metrics.get("f1_positive"),
                "ROC-AUC": metrics.get("roc_auc"),
                "PR-AUC": metrics.get("pr_auc"),
                "CV best": (evaluation.get("training_setup") or {}).get("cv_best_score"),
                "Calibración": ((evaluation.get("training_setup") or {}).get("calibration") or {}).get("method"),
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(by=["F1 positivo", "Recall positivo", "Accuracy"], ascending=False, na_position="last")
    return frame


def format_metric(value) -> str:
    return f"{value:.3f}" if value is not None else "N/D"


def display_curve(curve_path: str | None, caption: str, width: int = 320) -> None:
    resolved_curve = resolve_path(curve_path) if curve_path else None
    if resolved_curve and resolved_curve.exists():
        st.image(str(resolved_curve), caption=caption, width=width)
    else:
        st.info(f"{caption} no disponible")


@st.cache_resource(show_spinner=False)
def load_predictor_cached(checkpoint_path: str, metadata_path: str) -> Predictor:
    return Predictor(checkpoint_path=checkpoint_path, metadata_path=metadata_path)


@st.cache_resource(show_spinner=False)
def load_tabular_predictor_cached(checkpoint_path: str, metadata_path: str) -> TabularPredictor:
    return TabularPredictor(checkpoint_path=checkpoint_path, metadata_path=metadata_path)


def render_metric_help(metric_help: dict[str, str]) -> None:
    for metric_name, description in metric_help.items():
        st.markdown(f"**{metric_name}**: {description}")


def render_image_overview(models: list[dict], selected_model: dict) -> None:
    st.subheader("Modelos de imagen")
    frame = image_comparison_dataframe(models)
    if frame.empty:
        st.info("Todavía no hay modelos de imagen evaluados.")
        return

    with st.expander("Tabla comparativa de modelos de imagen", expanded=True):
        st.dataframe(frame, use_container_width=True, hide_index=True)
    with st.expander("Qué significa cada métrica", expanded=False):
        render_metric_help(IMAGE_METRIC_HELP)
    with st.expander("Gráfico comparativo entre modelos de imagen", expanded=True):
        st.bar_chart(frame.set_index("Modelo")[["F1 macro", "Recall macro", "Accuracy"]], use_container_width=True)

    evaluation = selected_model.get("evaluation")
    if not evaluation:
        st.info("El modelo seleccionado de imagen aún no tiene evaluación.")
        return
    metrics = evaluation["metrics"]
    with st.expander("Métricas del modelo de imagen seleccionado", expanded=True):
        cols = st.columns(5)
        cols[0].metric("Accuracy", format_metric(metrics.get("accuracy")))
        cols[1].metric("Recall macro", format_metric(metrics.get("recall_macro", metrics.get("recall"))))
        cols[2].metric("F1 macro", format_metric(metrics.get("f1_macro", metrics.get("f1"))))
        cols[3].metric("ROC-AUC", format_metric(metrics.get("roc_auc")))
        cols[4].metric("PR-AUC", format_metric(metrics.get("pr_auc")))

    if evaluation.get("training_setup"):
        with st.expander("Condiciones del entrenamiento de imagen", expanded=False):
            st.json(evaluation["training_setup"], expanded=False)
    if metrics.get("per_class"):
        with st.expander("Métricas por clase de imagen", expanded=False):
            class_frame = pd.DataFrame(
                [
                    {
                        "Clase": class_name,
                        "Precision": values.get("precision"),
                        "Recall": values.get("recall"),
                        "F1": values.get("f1"),
                        "Soporte": values.get("support"),
                    }
                    for class_name, values in metrics["per_class"].items()
                ]
            )
            st.dataframe(class_frame, use_container_width=True, hide_index=True)
    curves = evaluation.get("curve_paths", {})
    with st.expander("Curvas y matriz del modelo de imagen", expanded=False):
        left, right = st.columns(2)
        with left:
            display_curve(curves.get("roc_curve"), "Curva ROC")
            display_curve(curves.get("pr_curve"), "Curva Precision-Recall")
        with right:
            display_curve(curves.get("confusion_matrix_plot"), "Matriz de confusión", width=420)
            display_curve(curves.get("calibration_curve"), "Curva de calibración")


def render_tabular_overview(models: list[dict], selected_model: dict) -> None:
    st.subheader("Modelos tabulares")
    frame = tabular_comparison_dataframe(models)
    if frame.empty:
        st.info("Todavía no hay modelos tabulares evaluados.")
        return

    with st.expander("Tabla comparativa de modelos tabulares", expanded=True):
        st.dataframe(frame, use_container_width=True, hide_index=True)
    with st.expander("Qué significa cada métrica tabular", expanded=False):
        render_metric_help(TABULAR_METRIC_HELP)
    with st.expander("Gráfico comparativo entre modelos tabulares", expanded=True):
        st.bar_chart(frame.set_index("Modelo")[["F1 positivo", "Recall positivo", "Accuracy"]], use_container_width=True)

    evaluation = selected_model.get("evaluation")
    if not evaluation:
        st.info("El modelo tabular seleccionado aún no tiene evaluación.")
        return
    metrics = evaluation["metrics"]
    with st.expander("Métricas del modelo tabular seleccionado", expanded=True):
        cols = st.columns(6)
        cols[0].metric("Accuracy", format_metric(metrics.get("accuracy")))
        cols[1].metric("Precision positiva", format_metric(metrics.get("precision_positive")))
        cols[2].metric("Recall positivo", format_metric(metrics.get("recall_positive")))
        cols[3].metric("F1 positivo", format_metric(metrics.get("f1_positive")))
        cols[4].metric("ROC-AUC", format_metric(metrics.get("roc_auc")))
        cols[5].metric("PR-AUC", format_metric(metrics.get("pr_auc")))
    if evaluation.get("training_setup"):
        with st.expander("Condiciones del entrenamiento tabular", expanded=False):
            st.json(evaluation["training_setup"], expanded=False)
    if evaluation.get("cv_results_summary"):
        with st.expander("Resumen de búsqueda de hiperparámetros", expanded=False):
            st.write("Aquí se muestran las mejores combinaciones encontradas en validación cruzada.")
            st.dataframe(pd.DataFrame(evaluation["cv_results_summary"]), use_container_width=True, hide_index=True)
    if evaluation.get("feature_importance"):
        with st.expander("Variables más influyentes", expanded=False):
            importance_frame = pd.DataFrame(evaluation["feature_importance"])
            st.dataframe(importance_frame, use_container_width=True, hide_index=True)
            st.bar_chart(importance_frame.set_index("feature"))
    if evaluation.get("permutation_importance"):
        with st.expander("Importancia por permutación", expanded=False):
            st.write("Esta importancia es más robusta porque mide cuánto empeora el modelo si se desordena cada variable.")
            permutation_frame = pd.DataFrame(evaluation["permutation_importance"])
            st.dataframe(permutation_frame, use_container_width=True, hide_index=True)
            st.bar_chart(permutation_frame.set_index("feature"))
    curves = evaluation.get("curve_paths", {})
    with st.expander("Curvas y matriz del modelo tabular", expanded=False):
        left, right = st.columns(2)
        with left:
            display_curve(curves.get("roc_curve"), "Curva ROC")
            display_curve(curves.get("pr_curve"), "Curva Precision-Recall")
        with right:
            display_curve(curves.get("confusion_matrix_plot"), "Matriz de confusión", width=420)
            display_curve(curves.get("calibration_curve"), "Curva de calibración")
    if evaluation.get("source_alerts"):
        with st.expander("Alertas por subgrupo tabular", expanded=False):
            st.dataframe(pd.DataFrame(evaluation["source_alerts"]), use_container_width=True, hide_index=True)


def compute_image_prediction(selected_model: dict, image: Image.Image) -> dict:
    predictor = load_predictor_cached(
        checkpoint_path=str(selected_model["checkpoint_path"]),
        metadata_path=str(selected_model["metadata_path"]),
    )
    temp_path = IMAGE_ARTIFACTS_ROOT / "tmp_streamlit_image.png"
    image.save(temp_path)
    result = predictor.predict(temp_path)
    result["overlay"] = overlay_heatmap(image, result["heatmap"])
    return result


def render_image_prediction_result(result: dict) -> None:
    with st.expander("Resultado de imagen", expanded=True):
        predicted_label = result["predicted_class"].replace("_", " ").upper()
        cols = st.columns(3)
        cols[0].metric("Clase predicha", predicted_label)
        cols[1].metric("Confianza", f"{result['confidence']:.1%}")
        top_other = max((value for key, value in result["probabilities"].items() if key != result["predicted_class"]), default=0.0)
        cols[2].metric("Diferencia frente a la siguiente", f"{(result['confidence'] - top_other):.1%}")
        st.success(
            f"La imagen se clasifica como **{result['predicted_class'].replace('_', ' ')}** con una confianza aproximada del **{result['confidence']:.1%}**."
        )
    with st.expander("Grad-CAM y probabilidades de imagen", expanded=True):
        image_cols = st.columns(2)
        image_cols[0].image(result["overlay"], caption="Mapa Grad-CAM", use_container_width=True)
        probs = pd.DataFrame({"Clase": list(result["probabilities"].keys()), "Probabilidad": list(result["probabilities"].values())}).sort_values(by="Probabilidad", ascending=False)
        image_cols[1].bar_chart(probs.set_index("Clase"))
        image_cols[1].dataframe(probs, use_container_width=True, hide_index=True)


def render_tabular_form(question_specs: list[dict], key_prefix: str = "tabular") -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    with st.form(f"{key_prefix}_form"):
        st.write("Introduce los datos clínicos del paciente.")
        for spec in question_specs:
            variable = spec["variable"]
            question = spec["question"]
            if variable == "age":
                values[variable] = st.number_input(question, min_value=18, max_value=110, value=60, step=1, key=f"{key_prefix}_{variable}")
            else:
                labels = [option["label"] for option in spec["options"]]
                selected_label = st.selectbox(question, options=labels, key=f"{key_prefix}_{variable}")
                selected_option = next(option for option in spec["options"] if option["label"] == selected_label)
                values[variable] = selected_option["value"]
        submitted = st.form_submit_button("Calcular predicción tabular")
    return values if submitted else {}


def compute_tabular_prediction(selected_model: dict, patient_payload: dict) -> dict:
    predictor = load_tabular_predictor_cached(
        checkpoint_path=str(selected_model["checkpoint_path"]),
        metadata_path=str(selected_model["metadata_path"]),
    )
    return predictor.predict(patient_payload)


def render_tabular_prediction_result(result: dict) -> None:
    risk_probability = result["probabilities"].get("riesgo_clinico", 0.0)
    with st.expander("Resultado tabular", expanded=True):
        cols = st.columns(3)
        cols[0].metric("Clase predicha", result["predicted_class"].replace("_", " ").upper())
        cols[1].metric("Confianza", f"{result['confidence']:.1%}")
        cols[2].metric("Probabilidad de riesgo clínico", f"{risk_probability:.1%}")
        st.info(
            "La salida tabular representa un **riesgo clínico tabular**. "
            "Un valor positivo puede ser compatible con pólipo o con otras patologías, pero no diferencia entre ambas."
        )
    with st.expander("Probabilidades tabulares", expanded=True):
        probs = pd.DataFrame({"Clase": list(result["probabilities"].keys()), "Probabilidad": list(result["probabilities"].values())}).sort_values(by="Probabilidad", ascending=False)
        st.bar_chart(probs.set_index("Clase"))
        st.dataframe(probs, use_container_width=True, hide_index=True)


def combined_interpretation(image_result: dict, tabular_result: dict) -> tuple[str, str]:
    image_class = image_result["predicted_class"]
    risk_probability = tabular_result["probabilities"].get("riesgo_clinico", 0.0)
    high_risk = risk_probability >= 0.5
    if image_class == "sano" and high_risk:
        return (
            "Concordancia parcial con alerta",
            "La imagen sugiere tejido sano, pero los datos tabulares muestran riesgo clínico elevado. Conviene revisar el caso con más detalle.",
        )
    if image_class in {"polipo", "otras_patologias"} and high_risk:
        return (
            "Concordancia alta",
            "La imagen apunta a una clase patológica y el modelo tabular también señala riesgo clínico. La señal conjunta es consistente.",
        )
    if image_class in {"polipo", "otras_patologias"} and not high_risk:
        return (
            "Concordancia baja",
            "La imagen apunta a patología, pero la señal tabular es baja. La clase de imagen no se rebaja, aunque conviene interpretar el caso con cautela.",
        )
    return (
        "Concordancia favorable",
        "La imagen sugiere tejido sano y el modelo tabular no detecta riesgo clínico elevado. Ambas señales son coherentes.",
    )


def main() -> None:
    st.set_page_config(page_title="Predicción multimodal", layout="wide")
    st.title("Predicción académica multimodal para pólipos y patologías relacionadas")
    st.caption("Herramienta educativa. La imagen decide la clase final y los datos tabulares añaden contexto de riesgo clínico.")

    image_models = discover_image_models()
    tabular_models = discover_tabular_models()
    question_specs = build_question_specs(TABULAR_CONFIG, load_questions(TABULAR_CONFIG))

    if not image_models:
        st.error("No se han encontrado modelos de imagen en Predictor_models/artifacts/metrics.")
        return

    default_image_index = next((index for index, model in enumerate(image_models) if model["model_name"] == DEFAULT_IMAGE_MODEL), 0)
    selected_image_label = st.selectbox("Modelo de imagen", [model["label"] for model in image_models], index=default_image_index)
    selected_image_model = next(model for model in image_models if model["label"] == selected_image_label)

    selected_tabular_model = None
    if tabular_models:
        default_tabular_index = next((index for index, model in enumerate(tabular_models) if model["model_name"] == DEFAULT_TABULAR_MODEL), 0)
        selected_tabular_label = st.selectbox("Modelo tabular", [model["label"] for model in tabular_models], index=default_tabular_index)
        selected_tabular_model = next(model for model in tabular_models if model["label"] == selected_tabular_label)

    overview_tab, prediction_tab = st.tabs(["Comparativa", "Predicción"])

    with overview_tab:
        image_tab, tabular_tab = st.tabs(["Modelos de imagen", "Modelos tabulares"])
        with image_tab:
            render_image_overview(image_models, selected_image_model)
        with tabular_tab:
            if selected_tabular_model is None:
                st.info("Todavía no hay modelos tabulares guardados.")
            else:
                render_tabular_overview(tabular_models, selected_tabular_model)

    with prediction_tab:
        mode = st.radio("Modo de uso", options=["Solo imagen", "Solo datos tabulares", "Combinado"], horizontal=True)

        if mode == "Solo imagen":
            uploaded_image = st.file_uploader("Sube una imagen endoscópica", type=["jpg", "jpeg", "png"], key="image_only")
            if uploaded_image is not None:
                try:
                    image = load_image(uploaded_image)
                    image_result = compute_image_prediction(selected_image_model, image)
                    render_image_prediction_result(image_result)
                except Exception as exc:
                    st.error(f"No se pudo ejecutar la inferencia de imagen: {exc}")

        elif mode == "Solo datos tabulares":
            if selected_tabular_model is None:
                st.warning("No hay modelos tabulares disponibles todavía.")
            else:
                payload = render_tabular_form(question_specs, key_prefix="tabular_only")
                if payload:
                    try:
                        tabular_result = compute_tabular_prediction(selected_tabular_model, payload)
                        render_tabular_prediction_result(tabular_result)
                    except Exception as exc:
                        st.error(f"No se pudo ejecutar la inferencia tabular: {exc}")

        else:
            if selected_tabular_model is None:
                st.warning("Para el modo combinado hace falta al menos un modelo tabular entrenado.")
            else:
                uploaded_image = st.file_uploader("Sube una imagen endoscópica", type=["jpg", "jpeg", "png"], key="combined_image")
                payload = render_tabular_form(question_specs, key_prefix="combined")
                if uploaded_image is not None and payload:
                    try:
                        image = load_image(uploaded_image)
                        image_result = compute_image_prediction(selected_image_model, image)
                        tabular_result = compute_tabular_prediction(selected_tabular_model, payload)

                        image_col, tabular_col = st.columns(2)
                        with image_col:
                            render_image_prediction_result(image_result)
                        with tabular_col:
                            render_tabular_prediction_result(tabular_result)

                        title, explanation = combined_interpretation(image_result, tabular_result)
                        with st.expander("Interpretación combinada", expanded=True):
                            st.success(title)
                            st.write(explanation)
                    except Exception as exc:
                        st.error(f"No se pudo ejecutar la predicción combinada: {exc}")


if __name__ == "__main__":
    main()
