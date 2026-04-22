# Documentación del Sistema Predictor (v1: Expertos Binarios)

Este documento detalla la arquitectura, el flujo de datos y la organización del sistema de clasificación oncológica desarrollado en la fase inicial (**v1**).

## 1. Filosofía del Sistema (v1)
A diferencia de un modelo multiclase único, la **v1** utiliza un **Ensamble de Especialistas Binarios**. Se entrenan modelos independientes para cada hallazgo clínico, optimizando la sensibilidad para cada patología por separado.

### Modelos Disponibles:
1.  **Especialista en Pólipos**: Detecta tejido polipoideo frente a tejido sano u otras patologías.
2.  **Especialista en Sangre**: Detecta sangrado activo.
3.  **Especialista en Inflamación**: Detecta mucosa inflamada/eritematosa.
4.  **Especialista en Casos Negativos**: Clasifica imágenes como "Normales/Sanas".

---

## 2. Flujo de Preprocesamiento y Datos
Para garantizar la calidad clínica, las imágenes pasan por un pipeline de limpieza antes del entrenamiento:

### A. Text Removal (AOT-GAN)
El endoscopio incrusta metadatos (texto blanco/verde) en el margen izquierdo. Para evitar que la IA aprenda a clasificar según el texto del hospital:
- **`preprocess_masks.py`**: Detecta automáticamente el texto y genera una máscara.
- **`preprocess_inpaint.py`**: Utiliza una red **AOT-GAN** para rellenar ese hueco con texturas realistas de mucosa colorrectal.

### B. Aumento de Datos (Transforms)
Se aplican rotaciones, ajustes de brillo/contraste y normalización específica para equipos de endoscopía Olympus/Pentax.

---

## 3. Estructura del Proyecto (v1)
Ubicación: `Predictor_models/pipeline/v1_expert_binary/`

| Archivo | Función |
| :--- | :--- |
| `config.py` | Variables globales, rutas y hiperparámetros. |
| `dataset.py` | Clase `EndoDataset` que implementa One-vs-Rest. |
| `models.py` | Arquitectura basada en **EfficientNet-B0**. |
| `train.py` | Bucle de entrenamiento integrado con **MLflow**. |
| `run_full_pipeline.py` | Orquestador que entrena los 4 modelos uno tras otro vaciando la VRAM. |
| `preprocess_*.py` | Scripts de limpieza de imágenes. |

---

## 4. Resumen del Dataset
El sistema se nutre de aproximadamente **71,159 imágenes** totales, organizadas así:

| Categoría | Imágenes | Descripción |
| :--- | :--- | :--- |
| **Negativos** | ~1,500 | Casos de control sano. |
| **Pólipos** | 1,798 | Dataset consolidado con Ground Truth. |
| **Sin Clasificar** | ~61k | Banco de imágenes para entrenamiento de AOT-GAN. |

---

## 5. Monitorización (MLflow)
Todos los experimentos se registran en el experimento maestro: **`Predictor_Cancer_Endoscopico`**.
Para visualizar las métricas y descargar los modelos `.pth`:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
*(Acceder vía navegador a http://127.0.0.1:5000)*
