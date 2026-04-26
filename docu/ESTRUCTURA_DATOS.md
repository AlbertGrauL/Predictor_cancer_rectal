# Estructura del Dataset de Imágenes y Datos Clínicos

Este documento es la referencia definitiva sobre la organización de los datos en el proyecto. Detalla la jerarquía para el entrenamiento de los modelos de clasificación (v1/v2), el sistema de inpainting (AOT-GAN) y el análisis tabular.

> [!IMPORTANT]
> La carpeta Predictor_models/data/ está excluida de Git. Los datos deben seguir estrictamente esta estructura para asegurar la compatibilidad con los scripts de entrenamiento y validación.

---

## Jerarquía Global del Dataset

El dataset se organiza en bloques funcionales que separan el contenido clínico, el banco de imágenes crudas y los datos tabulares.

```text
Predictor_models/data/
├── tabulares/                           # -> Datos Clínicos Estructurados
│   └── cancer_final.csv                 # Base de datos de pacientes (anonymized)
│
├── imagenes_cancer/                     # -> Dataset de Imágenes Médicas
│   ├── Casos_negativos/                 # -> Control (Sano)
│   │   ├── normal-cecum/                # Mucosa del ciego
│   │   ├── normal-pylorus/              # Mucosa del píloro
│   │   └── dyed-resection-margins/      # Márgenes con tinte (post-resección)
│   │
│   ├── Polipos/                         # -> Neoplasias y Lesiones
│   │   ├── imagenes con polipos destacados/
│   │   │   ├── original/                # Imágenes seleccionadas (set1, set2, set3)
│   │   │   └── masks/                   # Máscaras de segmentación sincronizadas
│   │   └── polyps/                      # Dataset extra (ingesta v1)
│   │
│   ├── Sangre_Paredes/                  # -> Otras Patologías
│   │   ├── sangre_activa/               # Hemorragia luminal
│   │   └── inflamacion_leve/            # Eritema y friabilidad mucosa
│   │
│   └── imagenes sin clasificar/         # -> Banco Crudo (Big Data)
│       └── images/                      # >60,000 imágenes para entrenamiento AOT-GAN
│
└── [Generado]/                          # -> Carpetas de Proceso (No manuales)
    ├── text_masks/                      # Máscaras de texto técnico detectado
    └── aotgan_train/                    # Dataset de fine-tuning para reconstrucción
```

---

## Resumen Estadístico y Clínico

| Categoría Clínica | Caracterización Visual | Subcarpetas Clave | Estado |
| :--- | :--- | :--- | :---: |
| Negativos (Control) | Mucosa sana, patrones de Kudo I/II. | normal-cecum, normal-pylorus | Finalizado |
| Pólipos (Neoplasias) | Lesiones elevadas y planas (París Is/IIa). | original, masks, polyps | Finalizado |
| Sangre e Inflamación | Sangrado activo, colitis, eritema leve. | sangre_activa, inflamacion_leve | Expansión |
| Banco AOT-GAN | Texturas biológicas masivas. | imagenes sin clasificar/images | Finalizado |
| Datos Tabulares | Variables clínicas de pacientes. | tabulares/cancer_final.csv | Finalizado |

---

## Especificaciones de Integración Técnica

### 1. Sincronización Imagen-Máscara
En la carpeta Polipos/imagenes con polipos destacados/, la relación entre imagen y máscara es 1:1 mediante el nombre del archivo.
- Ejemplo: original/set1_0001.png <-> masks/set1_0001.png
- El script organizar_imagenes.py garantiza que no existan "huérfanos" (imágenes sin máscara o viceversa).

### 2. Flujo de Limpieza (Inpainting)
Para evitar el sesgo por texto en pantalla, el sistema genera versiones "limpias" de las imágenes:
1. Detección: Se generan máscaras en text_masks/.
2. Reconstrucción: El modelo AOT-GAN sustituye el texto por textura mucosa sintética, guardando el resultado en carpetas con el sufijo _cleaned.

### 3. Datos Tabulares
El archivo cancer_final.csv contiene la información clínica que alimenta los modelos de predicción basados en variables (no imágenes). Está diseñado para ser procesado por el pipeline de Predictor_models/pipeline/tabular_expert/.

---

## Mantenimiento del Dataset
Para regenerar o sincronizar la estructura tras añadir nuevos datos de pólipos, ejecuta:
```powershell
python Predictor_models/organizar_imagenes.py
```
(Este script procesa las fuentes crudas y las unifica bajo la carpeta Polipos/imagenes con polipos destacados/).
