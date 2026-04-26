# Manual de Usuario: Predictor de Cáncer Rectal

Este documento detalla los procedimientos para el entrenamiento de los modelos de IA y la puesta en marcha de la aplicación completa (API + Frontend).

---

## 1. Puesta en Marcha (Modo Aplicación)
Usa este modo si ya dispones de los pesos de los modelos (.pth y .pt) en la carpeta Predictor_models/artifacts/models/.

### Opción A: Lanzador Automático (Recomendado)
Ejecuta el archivo LAUNCHER.bat en la raíz del proyecto y selecciona la opción [3]:
- API: Se iniciará en http://127.0.0.1:8000.
- Frontend: Se iniciará en http://localhost:4200.

### Opción B: Comandos Manuales
Si prefieres usar la terminal, abre dos pestañas y ejecuta:

1. Servidor API (Python):
   uvicorn Predictor_api.main:app --reload --port 8000

2. Frontend (Angular):
   cd Predictor_front
   npm start

---

## 2. Entrenamiento de Modelos

### 2.1 Modelos de Clasificación Expertos (v1)
Estos modelos se encargan de detectar pólipos, sangre, inflamación y casos negativos. Se entrenan de forma local aprovechando la arquitectura EfficientNet-B0.

Comando de entrenamiento completo:
python -m Predictor_models.pipeline.v1_expert_binary.run_full_pipeline

Este comando ejecutará de forma secuencial el entrenamiento de las 4 categorías, guardando los mejores pesos en la carpeta de artefactos.

### 2.2 Modelo de Inpainting (AOT-GAN)
Debido a la alta demanda de cómputo y memoria GPU para la reconstrucción de texturas, el modelo de limpieza de imágenes (inpainting) debe entrenarse en Kaggle.

Pasos para el re-entrenamiento:
1. Subir el dataset de imagenes sin clasificar/ a un Dataset de Kaggle.
2. Utilizar el notebook de entrenamiento AOT-GAN optimizado para P100/T4.
3. Una vez finalizado el entrenamiento, descargar el archivo de pesos del generador (ej: G0100000.pt).
4. Colocar el archivo descargado en Predictor_models/artifacts/models/.
5. Verificar que el nombre del archivo coincida con la constante INPAINT_WEIGHTS definida en Predictor_models/pipeline/v1_expert_binary/config.py.

---

## 3. Requisitos Previos e Instalación

### Backend (Python)
Es necesario tener instalado Python 3.10+ y las dependencias principales:
- Torch y Torchvision (para los modelos de IA).
- FastAPI y Uvicorn (para la API).
- Python-multipart (para la subida de archivos).

Comando de instalación:
pip install torch torchvision torchaudio fastapi uvicorn python-multipart opencv-python pillow numpy

### Frontend (Node.js)
Es necesario tener instalado Node.js LTS y Angular CLI:
1. Instalar Angular CLI globalmente: npm install -g @angular/cli
2. Instalar dependencias locales:
   cd Predictor_front
   npm install

---

## 4. Gestión de Datos
Para que el sistema funcione, la estructura de carpetas en Predictor_models/data/ debe seguir estrictamente lo definido en el documento ESTRUCTURA_DATOS.md.

Si se añaden nuevas imágenes crudas para el dataset de pólipos, se debe sincronizar la estructura ejecutando:
python Predictor_models/organizar_imagenes.py
