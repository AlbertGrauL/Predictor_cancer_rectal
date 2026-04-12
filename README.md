# Predictor de Polipos y Patologias Relacionadas con Cancer Rectal

Proyecto academico para clasificacion de imagenes endoscopicas centrado en una primera fase binaria `polipo vs no_polipo`, con una arquitectura preparada para extenderse a un escenario multiclase con otras patologias gastrointestinales.

## Objetivos

- Entrenar y comparar varias CNN con transferencia de aprendizaje.
- Auditar y documentar el dataset de forma reproducible.
- Evaluar el rendimiento con metricas solidas y analisis de errores.
- Explicar las predicciones con Grad-CAM.
- Exponer el modelo en una interfaz simple con Streamlit.

## Estado del proyecto

La v1 esta orientada a un clasificador binario:

- `polipo`: `Predictor_models/data/imagenes_cancer/Polipos/polyps` + dataset consolidado `output/original`
- `no_polipo`: `Predictor_models/data/imagenes_cancer/Casos_negativos/*`

La carpeta `Sangre_Paredes` se reserva para evaluacion externa exploratoria y para una futura fase multiclase.

## Estructura

```text
Predictor_models/
  artifacts/              # Salidas de auditoria, metricas, figuras, pesos y metadatos
  configs/                # Configuracion YAML del pipeline
  data/                   # Dataset local (ignorado por Git)
  pipeline/               # Codigo de auditoria, preparacion, entrenamiento y evaluacion
Predictor_front/
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

- [ejecutar_pipeline_inicial.bat](/C:/Users/victo/Desktop/Predictor_cancer_rectal/ejecutar_pipeline_inicial.bat)
- [ejecutar_comparacion_modelos.bat](/C:/Users/victo/Desktop/Predictor_cancer_rectal/ejecutar_comparacion_modelos.bat)

El primero ejecuta:

- auditoria
- preparacion de splits
- entrenamiento del modelo base
- evaluacion del modelo base

El segundo ejecuta:

- auditoria
- preparacion de splits
- entrenamiento y evaluacion de cada CNN candidata
- resumen CSV comparativo final

Tambien puedes lanzarlos desde terminal:

```bash
.\ejecutar_pipeline_inicial.bat
.\ejecutar_comparacion_modelos.bat
```

### 1. Auditar el dataset

```bash
uv run python -m Predictor_models.pipeline.audit_dataset --config Predictor_models/configs/binary_baseline.yaml
```

Genera:

- manifiesto del dataset
- conteos por clase y fuente
- estadisticas de formatos y resoluciones
- deteccion basica de duplicados y archivos potencialmente corruptos
- informe en JSON y Markdown

### 2. Crear splits reproducibles

```bash
uv run python -m Predictor_models.pipeline.prepare_data --config Predictor_models/configs/binary_baseline.yaml
```

Genera:

- `dataset_manifest.csv`
- `splits.json`
- resumen de balance por split

### 3. Entrenar modelos

Ejemplo con EfficientNet-B0:

```bash
uv run python -m Predictor_models.pipeline.train --config Predictor_models/configs/binary_baseline.yaml --model efficientnet_b0
```

Modelos preparados:

- `resnet18`
- `resnet50`
- `efficientnet_b0`
- `efficientnet_b2`
- `densenet121`
- `convnext_tiny`

### 4. Evaluar un checkpoint

```bash
uv run python -m Predictor_models.pipeline.evaluate --config Predictor_models/configs/binary_baseline.yaml --checkpoint Predictor_models/artifacts/checkpoints/efficientnet_b0_best.pt
```

Genera:

- metricas globales
- matriz de confusion
- curvas ROC y PR
- calibracion
- analisis por fuente
- evaluacion externa exploratoria sobre `Sangre_Paredes`

### 5. Lanzar la app de Streamlit

```bash
uv run streamlit run Predictor_front/app.py
```

## Orquestadores Python

Si prefieres ejecutarlo sin los `.bat`, tambien tienes dos orquestadores:

- [run_initial_pipeline.py](/C:/Users/victo/Desktop/Predictor_cancer_rectal/Predictor_models/pipeline/run_initial_pipeline.py)
- [run_model_comparison.py](/C:/Users/victo/Desktop/Predictor_cancer_rectal/Predictor_models/pipeline/run_model_comparison.py)

Ejemplos:

```bash
uv run python -m Predictor_models.pipeline.run_initial_pipeline
uv run python -m Predictor_models.pipeline.run_initial_pipeline --model resnet50
uv run python -m Predictor_models.pipeline.run_model_comparison
uv run python -m Predictor_models.pipeline.run_model_comparison --models resnet50 efficientnet_b0 densenet121
```

## Metricas principales

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Matriz de confusion
- Calibracion basica

La seleccion del mejor modelo no debe hacerse solo por accuracy. En este proyecto se prioriza especialmente el `recall` y `F1` de la clase `polipo`.

## Interpretabilidad

La app y el pipeline soportan:

- probabilidad por clase
- prediccion top-1
- Grad-CAM sobre la region mas influyente

## Documentacion academica

La memoria tecnica completa esta en [docs/memoria_tecnica.md](/C:/Users/victo/Desktop/Predictor_cancer_rectal/docs/memoria_tecnica.md).

## Limitaciones y aviso

- Herramienta academica y de investigacion.
- No es un sistema de apoyo clinico validado.
- Las conclusiones deben interpretarse dentro de las limitaciones del dataset y del contexto docente.
