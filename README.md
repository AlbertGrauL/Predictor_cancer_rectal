# Predictor de Cáncer Colorrectal - Pipeline de Deep Learning

Este repositorio contiene un sistema modular basado en PyTorch para el entrenamiento y evaluación de modelos de clasificación de imágenes endoscópicas. El sistema está diseñado para entrenar 4 modelos especialistas: Pólipos, Sangre, Inflamación y Negativos.

## Puesta en Marcha

### 1. Requisitos e Instalación

Se recomienda usar un entorno de **Conda** para gestionar las dependencias y el soporte de GPU (NVIDIA).

```bash
# Activa tu entorno de conda (ej: inpainting)
conda activate inpainting

# Instala las dependencias necesarias
pip install torch torchvision torchaudio mlflow scikit-learn tqdm opencv-python pillow
```

### 2. Configuración de Rutas
Antes de entrenar, verifica que las rutas a las imágenes en `Predictor_models/pipeline/config.py` sean correctas para tu máquina local.

### 3. Entrenamiento de los Modelos
Para entrenar las 4 redes neuronales de forma secuencial y automática, utiliza el orquestador global. Este script gestiona la limpieza de memoria VRAM entre modelos para evitar errores de memoria.

```bash
# Ejecutar desde la raíz del proyecto
python -m Predictor_models.pipeline.run_full_pipeline
```

---

## Monitorización con MLflow

El proyecto integra **MLflow** para el seguimiento de experimentos. Registra automáticamente:
- **Hiperparámetros**: Learning rate, epochs, batch size.
- **Métricas por época**: Loss, AUC-ROC, Sensibilidad (Recall) y Especificidad.
- **Artefactos**: El mejor modelo `.pth` de cada entrenamiento.

### Cómo ver los resultados (Dashboard)
Para abrir la interfaz visual de MLflow y ver las gráficas de entrenamiento:

1. Abre una **nueva terminal** de Conda.
2. Activa tu entorno.
3. Ejecuta el siguiente comando indicando la base de datos local:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

4. Abre tu navegador en: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Estructura del Pipeline (`Predictor_models/pipeline/`)
- `config.py`: Configuración centralizada.
- `dataset.py`: Cargador de datos (One-vs-Rest).
- `models.py`: Arquitecturas (EfficientNet-B0).
- `train.py`: Bucle de entrenamiento con hooks de MLflow.
- `metrics.py`: Cálculo de métricas clínicas.
- `transforms.py`: Aumento de datos para endoscopía.
- `run_full_pipeline.py`: Script maestro de entrenamiento secuencial.

---

## Herramientas de Preprocesamiento (AOT-GAN)
Si deseas realizar la eliminación de texto (Inpainting) antes del entrenamiento:
1. Prepara el dataset limpio: `python -m Predictor_models.pipeline.aotgan_prepare`
2. Genera las máscaras de texto: `python -m Predictor_models.pipeline.preprocess_masks`
3. Aplica la limpieza (requiere modelo AOT-GAN entrenado): `python -m Predictor_models.pipeline.preprocess_inpaint`

---

## 📁 Dataset y Estructura de Datos

El sistema espera que las imágenes estén organizadas en la carpeta `Predictor_models/data/imagenes_cancer/`. A continuación se detalla la estructura principal:

```text
imagenes_cancer/
├── Casos_negativos/         # Control negativo (resección, ciego, píloro) (~1500 img)
├── Polipos/                 # Dataset consolidado de pólipos
│   └── output/
│       ├── original/        # 1,798 imágenes unificadas
│       └── masks/           # Máscaras sincronizadas
├── Sangre_Paredes/          # Categorías de inflamación y sangre activa
└── imagenes sin clasificar/ # Banco de imágenes crudas (~61k img)
```

### Resumen de Datos Disponibles
| Categoría | Carpeta | Imágenes |
| :--- | :--- | :--- |
| **Negativos** | `Casos_negativos/` | ~1,500 |
| **Pólipos (Consolidado)** | `Polipos/.../output/original` | 1,798 |
| **Sangre / Inflamación** | `Sangre_Paredes/` | (En proceso) |
| **Sin Clasificar** | `imagenes sin clasificar/` | ~61,957 |
| **TOTAL** | | **~71,159** |

---

## Ejecución de la Aplicación (API y Frontend)

Para poner en marcha la aplicación completa (Back-end y Front-end), sigue estos pasos:

### 1. Iniciar el Back-end (API)

El servidor de API usa FastAPI. Para iniciarlo, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
# Con uv (recomendado)
uv run uvicorn Predictor_api.main:app --reload

# O directamente con uvicorn si tienes el entorno activado
uvicorn Predictor_api.main:app --reload
```

La API estará disponible en: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### 2. Iniciar el Front-end (Angular)

El front-end está desarrollado en Angular. Navega a la carpeta correspondiente e inicia el servidor:

```bash
cd Predictor_front
npm install  # Solo la primera vez
npm start
```

La aplicación web estará disponible en: [http://localhost:4200](http://localhost:4200)

---

## Resultados

Esta sección compara los dos enfoques de modelado implementados en el proyecto.

---

### V1 — Especialistas Binarios (`Predictor_models`)

El primer enfoque entrena **4 modelos independientes** bajo la estrategia *One-vs-Rest*: cada red aprende a detectar su categoría clínica frente a todas las demás.

**Configuración de entrenamiento:**

| Parámetro | Valor |
| :--- | :--- |
| Arquitectura | EfficientNet-B0 (ImageNet pretrained) |
| Estrategia | One-vs-Rest (4 clasificadores binarios) |
| Optimizer | Adam |
| Loss | BCEWithLogitsLoss |
| Split Train/Val/Test | 80 / 10 / 10 % |
| Early stopping | patience = 7 |
| Dataset total | ~4 900 imágenes (train+val) por especialista |

**Mejores métricas de validación** *(mejor checkpoint por AUC-ROC)*:

| Especialista | Val Loss | AUC-ROC | Sensibilidad |
| :--- | :---: | :---: | :---: |
| Pólipos | 0.0008 | **1.0000** | **1.0000** |
| Inflamación | 0.0727 | **0.9930** | 0.8537 |
| Sangre | 0.0753 | 0.9897 | 0.9412 |
| Negativos | 0.4224 | 0.7177 | 0.6029 |

> **Observación:** El modelo de *Negativos* muestra un AUC considerablemente inferior (0.72). La categoría agrupa imágenes heterogéneas (ciego normal, píloro, márgenes de resección) que comparten rasgos visuales con las categorías positivas, lo que dificulta la separación binaria.

---

### V2 — Clasificador Multiclase con Comparativa de Arquitecturas (`predictor_models_v`)

El segundo enfoque unifica el problema en un único clasificador que distingue entre tres clases: **polipo**, **sano** y **otras_patologias**. Se comparan tres arquitecturas de forma sistemática con el mismo pipeline.

**Configuración de entrenamiento:**

| Parámetro | Valor |
| :--- | :--- |
| Arquitecturas comparadas | ResNet-50 · EfficientNet-B0 · DenseNet-121 |
| Estrategia | Multiclase unificada (3 clases) |
| Optimizer | AdamW (lr = 3 × 10⁻⁴, wd = 1 × 10⁻⁴) |
| Scheduler | ReduceLROnPlateau (factor 0.5, patience 2) |
| Loss | CrossEntropyLoss con pesos de clase |
| Split Train/Val/Test | 70 / 15 / 15 % |
| Early stopping | patience = 5 |
| Fine-tuning | Freeze backbone (época 1) → descongelado completo (época 2+) |

**Preprocesamiento y aumentación añadidos:**

| Técnica | Descripción |
| :--- | :--- |
| Bottom-left mask | Enmascara el 30 % × 35 % inferior-izquierdo (texto de equipo) |
| Random Erasing | p = 0.25, scale [0.02–0.10] (regularización) |

**Métricas primarias reportadas:** Accuracy · F1-macro · F1-weighted · AUC-ROC · PR-AUC

> **Estado:** Los artefactos de V2 aún no han sido generados. Para ejecutar la comparativa completa: `python -m predictor_models_v.pipeline.image.run_model_comparison`

---

### Comparativa de Enfoques

| Aspecto | V1 — OvR Binario | V2 — Multiclase |
| :--- | :--- | :--- |
| Nº de modelos | 4 (uno por clase) | 1 (arquitecturas rotadas) |
| Salida | Probabilidad binaria por clase | Softmax sobre 3 clases |
| Clases | pólipos / sangre / inflamación / negativos | polipo / sano / otras_patologias |
| Optimizer | Adam | AdamW + weight decay |
| Loss | BCEWithLogitsLoss | CrossEntropyLoss + class weights |
| Scheduler | — | ReduceLROnPlateau |
| Preprocessing | — | Máscara esquina + Random Erasing |
| Fine-tuning | Backbone completo desde epoch 1 | Freeze → unfreeze en epoch 2 |
| Métricas objetivo | AUC-ROC, Sensibilidad | F1-macro, Accuracy, AUC-ROC, PR-AUC |
| Arquitecturas probadas | EfficientNet-B0 | ResNet-50, EfficientNet-B0, DenseNet-121 |

**Ventajas de V2 frente a V1:**
- Elimina la inconsistencia del modelo *Negativos* al modelar todas las clases conjuntamente.
- Los pesos de clase en la loss compensan el desbalance sin necesidad de submuestreo.
- La comparativa sistemática de arquitecturas permite seleccionar la óptima con datos reales.
- El preprocesamiento de la máscara inferior-izquierda elimina texto de equipo que actúa como atajo (*shortcut*) en V1.

