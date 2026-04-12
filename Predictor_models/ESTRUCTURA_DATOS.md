# 📁 Estructura del Dataset de Imágenes

> **Importante:** La carpeta `imagenes_cancer/` está excluida de Git debido al volumen
> de imágenes. Descarga los datos por separado y colócalos siguiendo esta estructura.

---

## Árbol de carpetas

```
imagenes_cancer/
│
├── Casos_negativos/                          # Imágenes sin pólipos (control negativo)
│   ├── dyed-resection-margins/               # Márgenes de resección teñidos
│   │   └── *.jpg                             →  500 imágenes
│   ├── normal-cecum/                         # Ciego normal
│   │   └── *.jpg                             →  500 imágenes
│   └── normal-pylorus/                       # Píloro normal
│       └── *.jpg                             →  500 imágenes
│
├── Polipos/                                  # Imágenes con pólipos
│   ├── polyps/                               # Pólipos (dataset externo, UUID)
│   │   └── *.jpg                             →  500 imágenes
│   │
│   └── imagenes con polipos destacados/      # Dataset principal con segmentación
│       ├── Original/                         # Imágenes endoscópicas originales
│       │   └── *.png                         →  612 imágenes
│       ├── Ground Truth/                     # Máscaras binarias (par de Original/)
│       │   └── *.png                         →  612 imágenes
│       ├── images/                           # Segunda colección de imágenes
│       │   └── *.png                         →  196 imágenes
│       ├── masks/                            # Máscaras binarias (par de images/)
│       │   └── *.png                         →  196 imágenes
│       ├── images_2/                         # Tercera colección (nombres hash)
│       │   └── *.jpg                         →  990 imágenes
│       ├── masks_2/                          # Máscaras binarias (par de images_2/)
│       │   └── *.jpg                         → 1000 imágenes
│       ├── bounding-boxes.json               # Anotaciones de bounding boxes
│       │
│       └── output/                           ⭐ DATASET CONSOLIDADO (listo para entrenar)
│           ├── original/                     # Imágenes endoscópicas unificadas
│           │   └── set{1,2,3}_NNNN.{png,jpg} → 1 798 imágenes
│           └── masks/                        # Máscaras sincronizadas (mismo nombre)
│               └── set{1,2,3}_NNNN.{png,jpg} → 1 798 imágenes
│
├── Sangre_Paredes/                           # Imágenes de sangre y paredes intestinales
│   ├── inflamacion_leve/                     # Inflamación leve (pendiente de imágenes)
│   │   └── (vacío)                           →    0 imágenes
│   └── sangre_activa/                        # Sangrado activo (pendiente de imágenes)
│       └── (vacío)                           →    0 imágenes
│
└── imagenes sin clasificar/                  # Imágenes sin categoría asignada
    └── images/
        └── *.jpg                             → 61 957 imágenes
```

---

## Resumen por categoría

| Categoría                          | Carpeta                            | Imágenes |
|------------------------------------|------------------------------------|--------:|
| Casos negativos — resección teñida | `Casos_negativos/dyed-resection-margins` |    500 |
| Casos negativos — ciego normal     | `Casos_negativos/normal-cecum`          |    500 |
| Casos negativos — píloro normal    | `Casos_negativos/normal-pylorus`        |    500 |
| Pólipos (dataset externo)          | `Polipos/polyps`                        |    500 |
| Pólipos segmentados — originales   | `Polipos/imagenes con polipos destacados/Original`      |    612 |
| Pólipos segmentados — ground truth | `Polipos/imagenes con polipos destacados/Ground Truth`  |    612 |
| Pólipos segmentados — images       | `Polipos/imagenes con polipos destacados/images`        |    196 |
| Pólipos segmentados — masks        | `Polipos/imagenes con polipos destacados/masks`         |    196 |
| Pólipos segmentados — images_2     | `Polipos/imagenes con polipos destacados/images_2`      |    990 |
| Pólipos segmentados — masks_2      | `Polipos/imagenes con polipos destacados/masks_2`       |  1 000 |
| **Dataset consolidado — original** | `…/output/original`                    | **1 798** |
| **Dataset consolidado — masks**    | `…/output/masks`                       | **1 798** |
| Sangre / inflamación leve          | `Sangre_Paredes/inflamacion_leve`       |      0 |
| Sangre / sangre activa             | `Sangre_Paredes/sangre_activa`          |      0 |
| Sin clasificar                     | `imagenes sin clasificar/images`        | 61 957 |
| **TOTAL**                          |                                         | **~71 159** |

---

## Dataset consolidado para entrenamiento

La carpeta `output/` fue generada por el script [`organizar_imagenes.py`](../organizar_imagenes.py).
Combina tres fuentes de datos con pares imagen–máscara perfectamente sincronizados:

| Prefijo     | Fuente                                | Pares  | Formato |
|-------------|---------------------------------------|-------:|---------|
| `set1_NNNN` | `Original/` ↔ `Ground Truth/`         |    612 | `.png`  |
| `set2_NNNN` | `images/` ↔ `masks/`                  |    196 | `.png`  |
| `set3_NNNN` | `images_2/` ↔ `masks_2/` (intersección) |  990 | `.jpg`  |
| **Total**   |                                       | **1 798** |      |

> Cada imagen en `output/original/` tiene exactamente una máscara con el **mismo
> nombre** en `output/masks/`, lo que garantiza la correcta carga por índice en
> cualquier framework de deep learning (PyTorch, TensorFlow, Keras…).

---

## Notas

- Las carpetas `Sangre_Paredes/inflamacion_leve` y `Sangre_Paredes/sangre_activa`
  están reservadas para imágenes futuras.
- `masks_2/` contiene 1 000 imágenes, pero solo 990 tienen su pareja en `images_2/`;
  los 10 huérfanos se excluyen automáticamente al consolidar.
- Para regenerar el dataset consolidado ejecuta:
  ```bash
  py Predictor_models/data/imagenes_cancer/Polipos/organizar_imagenes.py
  ```
