# V3 — Tabular Fusion

Pipeline de fusión de datos tabulares (clínicos) con la señal de imagen.

## Estructura

```
v3_tabular_fusion/
├── image/      # Módulos de inferencia de imagen (hereda de v1/v2)
├── tabular/    # Scorer clínico y modelo tabular (XGBoost / RF)
└── fusion.py   # Lógica de combinación imagen × clínica
```

## Estado

Carpeta en desarrollo. El scorer heurístico está implementado en `Predictor_api/main.py`.
Pendiente: entrenamiento del modelo tabular una vez disponible `cancer_final.csv`.
