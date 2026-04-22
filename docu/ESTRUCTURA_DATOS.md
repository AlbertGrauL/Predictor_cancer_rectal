# 📁 Estructura del Dataset de Imágenes Endoscópicas

Este documento detalla la organización de las imágenes para el entrenamiento de los modelos de clasificación (**v1**) y el sistema de inpainting (**AOT-GAN**).

> [!IMPORTANT]
> La carpeta `imagenes_cancer/` está excluida de Git. Asegúrate de colocar los datos siguiendo estrictamente esta estructura para que los scripts de entrenamiento funcionen sin errores de ruta.

---

## 🏗️ Jerarquía del Proyecto de Datos

El dataset se divide en tres bloques funcionales: **Diagnóstico (Clasificados)**, **Banco Crudo (Sin Clasificar)** y **Artefactos de Limpieza**.

```text
Predictor_models/data/
├── imagenes_cancer/                     # Dataset Principal
│   ├── Casos_negativos/                 # → 1,500 imgs (Ciego, Píloro, Resección)
│   ├── Polipos/                         # → Dataset segmentado y consolidado
│   │   └── output/
│   │       ├── original/                # 1,798 imgs (set1, set2, set3)
│   │       └── masks/                   # Máscaras de segmentación sincronizadas
│   ├── Sangre_Paredes/                  # → Sangrado e inflamación leve
│   └── imagenes sin clasificar/         # → 61,957 imgs (Banco para AOT-GAN)
│
├── text_masks/                          # [Generado] Máscaras de texto detectado
└── aotgan_train/                        # [Generado] Dataset para fine-tuning de AOT-GAN
    ├── images/                          # Recortes de mucosa limpia
    └── masks/                           # Máscaras sintéticas (random strokes)
```

---

## 📊 Resumen por Categoría Clínica

| Categoría Clinica | Caracterización de Visualización | Cantidad | Estado |
| :--- | :--- | :---: | :--- |
| **Negativos (Control)** | Mucosa colónica normal, patrones de Pit de Kudo Grado I/II. | ~1,500 | ✅ Listo |
| **Neoplasias (Pólipos)** | Lesiones planas (IIa), elevadas (Is) y pedunculadas (Ip). Clasificación de París. | 1,798 | ✅ Listo |
| **Hemorragia (Sangre)** | Sangrado luminal activo o coágulos adherentes; riesgo de lesión T1/T2. | - | ⏳ Captura |
| **Actividad Inflamatoria** | Eritema, pérdida de vasos y friabilidad mucosa (Puntaje Mayo 1-3). | - | ⏳ Captura |
| **Corpus AOT-GAN** | Dataset masivo para fine-tuning de reconstrucción de texturas biológicas. | 61,957 | ✅ Listo |

---

## 🛠️ Procesamiento y Consolidación

### 1. Consolidación de Pólipos
Las imágenes de pólipos provienen de diversas fuentes con distintos formatos. Se unifican mediante el script de organización:
- **Script**: `Predictor_models/organizar_imagenes.py`
- **Prefijos**: `set1_` (png), `set2_` (png), `set3_` (jpg).

### 2. Flujo de Limpieza (Inpainting)
Para evitar el sesgo por texto en pantalla, el sistema genera versiones "limpias" de las imágenes:
- **`text_masks/`**: Contiene máscaras binarias donde el blanco representa texto clínico.
- **`*_cleaned/`**: Carpetas generadas (ej. `Polipos_cleaned`) tras pasar por el modelo AOT-GAN, donde el texto ha sido sustituido por textura de mucosa sintética.

---

## 📝 Especificaciones de Integración Técnica

- **Coherencia Geométrica**: En la carpeta `Polipos/output`, cada frame y su correspondiente máscara binaria comparten una clave primaria (nombre de archivo). Esto permite cargar lotes de entrenamiento sin desajustes espaciales.
- **Filtrado de Huérfanos**: El proceso de consolidación detecta y segrega automáticamente registros incompletos, garantizando la integridad referencial del dataset.
- **Consolidación del Baseline**: Los prefijos `set1_`, `set2_` y `set3_` permiten la trazabilidad del origen del dato (datasets públicos vs capturas institucionales), facilitando estudios de generalización de dominio.
- **Mantenimiento**: Para regenerar la estructura consolidada en una nueva máquina:
  ```bash
  python Predictor_models/organizar_imagenes.py
  ```
