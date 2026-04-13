# Predictor Multiclase de Polipos y Patologias Relacionadas con Cancer Rectal

Proyecto academico para clasificacion multiclase de imagenes endoscopicas orientado a la prediccion `polipo / sano / otras_patologias`, centrado en tres CNN recomendadas para este problema.

## Objetivos

- Entrenar y comparar varias CNN con transferencia de aprendizaje.
- Auditar y documentar el dataset de forma reproducible.
- Evaluar el rendimiento con metricas solidas y analisis de errores.
- Explicar las predicciones con Grad-CAM.
- Exponer el modelo en una interfaz simple con Streamlit.

## Estado del proyecto

La configuracion activa por defecto esta orientada a un clasificador multiclase:

- `polipo`: `Predictor_models/data/imagenes_cancer/Polipos/polyps` + dataset consolidado `output/original`
- `sano`: `Predictor_models/data/imagenes_cancer/Casos_negativos/*`
- `otras_patologias`: `Predictor_models/data/imagenes_cancer/Sangre_Paredes/*`

Configuracion disponible:

- [multiclass_baseline.yaml](Predictor_models/configs/multiclass_baseline.yaml) como configuracion principal del proyecto

## Estructura

```text
Predictor_models/
  artifacts/              # Salidas de auditoria, metricas, figuras, pesos y metadatos
  configs/                # Configuracion YAML del pipeline
  data/                   # Dataset local (ignorado por Git)
  pipeline/               # Codigo de auditoria, preparacion, entrenamiento y evaluacion
Predictor_models/app/
  app.py                  # Aplicacion Streamlit
Predictor_api/
  README.md               # Espacio reservado para una futura API
docs/
  memoria_tecnica.md      # Documentacion academica del proyecto
```

## Requisitos recomendados

- Python `3.11` o `3.12`
- GPU opcional para acelerar entrenamiento

El entorno actual del repositorio usa Python `3.14.2`, pero algunas dependencias de deep learning pueden no tener soporte estable para esa version. Si `uv sync` falla, la recomendacion es recrear el entorno en Python 3.11 o 3.12.

## Instalacion

```bash
uv sync
```

Si fuera necesario recrear el entorno en una version compatible:

```bash
uv venv --python 3.12
uv sync
```

## Flujo de trabajo

### Lanzadores directos en Windows

Si quieres ejecutarlo sin escribir toda la secuencia a mano, en la raiz del proyecto tienes:

- [ejecutar_reentreno_base_multiclase.bat](ejecutar_reentreno_base_multiclase.bat)
- [ejecutar_reentreno_completo_multiclase.bat](ejecutar_reentreno_completo_multiclase.bat)

El primero ejecuta:

- auditoria
- preparacion de splits
- entrenamiento del modelo base
- evaluacion del modelo base

El segundo ejecuta:

- auditoria
- preparacion de splits
- entrenamiento y evaluacion de las tres CNN recomendadas
- resumen CSV comparativo final

Tambien puedes lanzarlos desde terminal:

```bash
.\ejecutar_reentreno_base_multiclase.bat
.\ejecutar_reentreno_completo_multiclase.bat
```

### 1. Auditar el dataset

```bash
uv run python -m Predictor_models.pipeline.audit_dataset --config Predictor_models/configs/multiclass_baseline.yaml
```

Genera:

- manifiesto del dataset
- conteos por clase y fuente
- estadisticas de formatos y resoluciones
- deteccion basica de duplicados y archivos potencialmente corruptos
- informe en JSON y Markdown

### 2. Crear splits reproducibles

```bash
uv run python -m Predictor_models.pipeline.prepare_data --config Predictor_models/configs/multiclass_baseline.yaml
```

Genera:

- `dataset_manifest.csv`
- `splits.json`
- resumen de balance por split

### 3. Entrenar modelos

Ejemplo con EfficientNet-B0:

```bash
uv run python -m Predictor_models.pipeline.train --config Predictor_models/configs/multiclass_baseline.yaml --model efficientnet_b0
```

Modelos recomendados y activos en el proyecto:

- `resnet50`
- `efficientnet_b0`
- `densenet121`

### 4. Evaluar un checkpoint

```bash
uv run python -m Predictor_models.pipeline.evaluate --config Predictor_models/configs/multiclass_baseline.yaml --checkpoint Predictor_models/artifacts/checkpoints/efficientnet_b0_best.pt
```

Genera:

- metricas globales
- matriz de confusion
- curvas ROC y PR
- calibracion
- analisis por fuente
- metricas por clase
- comparativa multiclase

### 5. Lanzar la app de Streamlit

```bash
uv run streamlit run Predictor_models/app/app.py
```

## Orquestadores Python

Si prefieres ejecutarlo sin los `.bat`, tambien tienes dos orquestadores:

- [run_initial_pipeline.py](Predictor_models/pipeline/run_initial_pipeline.py)
- [run_model_comparison.py](Predictor_models/pipeline/run_model_comparison.py)

Ejemplos:

```bash
uv run python -m Predictor_models.pipeline.run_initial_pipeline
uv run python -m Predictor_models.pipeline.run_initial_pipeline --model resnet50
uv run python -m Predictor_models.pipeline.run_model_comparison
uv run python -m Predictor_models.pipeline.run_model_comparison --models resnet50 efficientnet_b0 densenet121
```

## Metricas principales

- Accuracy
- Precision macro
- Recall macro
- F1 macro
- F1 weighted
- ROC-AUC
- PR-AUC
- Matriz de confusion
- Metricas por clase

La seleccion del mejor modelo no debe hacerse solo por accuracy. En esta version multiclase se priorizan especialmente `F1 macro`, `Recall macro` y el comportamiento por clase.

## Interpretabilidad

La app y el pipeline soportan:

- probabilidad por clase
- prediccion top-1
- Grad-CAM sobre la region mas influyente

## Documentacion academica

La memoria tecnica completa esta en [docs/memoria_tecnica.md](docs/memoria_tecnica.md).

## Limitaciones y aviso

- Herramienta academica y de investigacion.
- No es un sistema de apoyo clinico validado.
- Las conclusiones deben interpretarse dentro de las limitaciones del dataset y del contexto docente.
