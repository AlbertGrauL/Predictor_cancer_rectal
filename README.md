# Predictor Multimodal de Polipos y Patologias Relacionadas con Cancer Rectal

Proyecto academico con dos vias de prediccion complementarias:

- `Imagen endoscopica`: clasificacion multiclase `polipo / sano / otras_patologias`
- `Datos tabulares de paciente`: clasificacion binaria `sin_riesgo_clinico / riesgo_clinico`

La app integra ambas salidas en tres modos:

- `Solo imagen`
- `Solo datos tabulares`
- `Combinado`

En el modo combinado, la imagen decide la clase final y el modelo tabular aporta apoyo de riesgo clinico.

## Objetivos

- Entrenar y comparar tres CNN recomendadas para imagen medica.
- Entrenar y comparar dos modelos tabulares clasicos: `RandomForest` y `XGBoost`.
- Auditar y preparar de forma reproducible tanto los datos de imagen como los tabulares.
- Evaluar los modelos con metricas, curvas, calibracion y analisis de errores.
- Explicar la salida de imagen mediante `Grad-CAM`.
- Mostrar todo en una app sencilla de Streamlit orientada a defensa academica.

## Modalidades del proyecto

### Imagen

Configuracion principal: [multiclass_baseline.yaml](Predictor_models/configs/image/multiclass_baseline.yaml)

Clases actuales:

- `polipo`
- `sano`
- `otras_patologias`

Modelos activos:

- `resnet50`
- `efficientnet_b0`
- `densenet121`

### Tabular

Configuracion principal: [tabular_baseline.yaml](Predictor_models/configs/tabular/tabular_baseline.yaml)

Objetivo:

- `cancer_diagnosis`: `no -> 0`, `yes -> 1`

Modelos activos:

- `random_forest`
- `xgboost`

Mejoras activas en esta iteracion:

- limpieza y normalizacion de variables
- validacion cruzada estratificada
- busqueda de hiperparametros con `RandomizedSearchCV`
- calibracion de probabilidades
- importancia de variables por atributo y por permutacion

## Estructura

```text
Predictor_models/
  app/                    # App Streamlit
  artifacts/              # Artefactos de imagen
  artifacts/tabular/      # Artefactos tabulares
  configs/image/          # Configuracion del flujo CNN
  configs/tabular/        # Configuracion del flujo tabular
  data/                   # Dataset local
  pipeline/image/         # Pipeline de clasificacion de imagenes
  pipeline/tabular/       # Pipeline de clasificacion tabular
  pipeline/               # Modulos compartidos
docs/
  memoria_tecnica.md      # Documentacion academica
```

## Requisitos

- Python `3.11` o `3.12` recomendado
- GPU opcional para entrenamiento de CNN

Instalacion:

```bash
uv sync
```

Si hiciera falta recrear el entorno:

```bash
uv venv --python 3.12
uv sync
```

## Lanzadores directos

En la raiz del proyecto tienes:

- [ejecutar_reentreno_base_multiclase.bat](ejecutar_reentreno_base_multiclase.bat)
- [ejecutar_reentreno_completo_multiclase.bat](ejecutar_reentreno_completo_multiclase.bat)
- [ejecutar_comparacion_tabular.bat](ejecutar_comparacion_tabular.bat)

## Flujo de imagen

### 1. Auditoria

```bash
uv run python -m Predictor_models.pipeline.image.audit_dataset --config Predictor_models/configs/image/multiclass_baseline.yaml
```

### 2. Preparacion de manifiesto y splits

```bash
uv run python -m Predictor_models.pipeline.image.prepare_data --config Predictor_models/configs/image/multiclass_baseline.yaml
```

### 3. Entrenamiento de una CNN

```bash
uv run python -m Predictor_models.pipeline.image.train --config Predictor_models/configs/image/multiclass_baseline.yaml --model efficientnet_b0
```

### 4. Evaluacion

```bash
uv run python -m Predictor_models.pipeline.image.evaluate --config Predictor_models/configs/image/multiclass_baseline.yaml --checkpoint Predictor_models/artifacts/checkpoints/efficientnet_b0_best.pt
```

### 5. Comparacion completa

```bash
uv run python -m Predictor_models.pipeline.image.run_model_comparison
```

## Flujo tabular

### 1. Auditoria tabular y del formulario

```bash
uv run python -m Predictor_models.pipeline.tabular.audit_tabular_data --config Predictor_models/configs/tabular/tabular_baseline.yaml
```

Genera:

- `tabular_dataset_audit.json`
- `tabular_question_specs.json`

### 2. Preparacion del manifiesto tabular

```bash
uv run python -m Predictor_models.pipeline.tabular.prepare_tabular_data --config Predictor_models/configs/tabular/tabular_baseline.yaml
```

Genera:

- `tabular_manifest.csv`
- `tabular_split_summary.json`

### 3. Entrenamiento de un modelo tabular

```bash
uv run python -m Predictor_models.pipeline.tabular.train_tabular --config Predictor_models/configs/tabular/tabular_baseline.yaml --model xgboost
```

Que hace ahora este entrenamiento:

- selecciona solo variables numericas codificadas
- ejecuta `RandomizedSearchCV` con validacion cruzada estratificada
- busca hiperparametros sobre `train`
- selecciona el mejor estimador por score de CV
- calibra probabilidades usando `val`
- guarda checkpoint, metadatos e historial

### 4. Evaluacion tabular

```bash
uv run python -m Predictor_models.pipeline.tabular.evaluate_tabular --config Predictor_models/configs/tabular/tabular_baseline.yaml --checkpoint Predictor_models/artifacts/tabular/checkpoints/xgboost_tabular.pkl
```

Genera:

- metricas globales
- ROC, PR y calibracion
- matriz de confusion
- alertas por subgrupo
- importancia de variables
- importancia por permutacion

### 5. Comparacion tabular completa

```bash
uv run python -m Predictor_models.pipeline.tabular.run_tabular_model_comparison
```

## App Streamlit

Lanzamiento:

```bash
uv run streamlit run Predictor_models/app/app.py
```

La app muestra:

- comparativa separada de modelos de imagen
- comparativa separada de modelos tabulares
- selector independiente de modelo por modalidad
- prediccion de imagen con `Grad-CAM`
- prediccion tabular con formulario guiado
- modo combinado con interpretacion conjunta

## Metricas principales

### Imagen

- Accuracy
- Recall macro
- F1 macro
- ROC-AUC
- PR-AUC
- metricas por clase

### Tabular

- Accuracy
- Precision positiva
- Recall positivo
- F1 positivo
- ROC-AUC
- PR-AUC
- calibracion
- CV best score

## Documentacion academica

La memoria tecnica completa esta en [docs/memoria_tecnica.md](docs/memoria_tecnica.md).

## Limitaciones

- Herramienta academica y de investigacion.
- No es un sistema de apoyo clinico validado.
- La salida tabular representa riesgo clinico, no una clase patologica especifica.
- La salida combinada es reglada e interpretable, no una fusion entrenada.
