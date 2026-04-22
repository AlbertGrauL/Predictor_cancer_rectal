# Analisis Historico de Metricas

Este archivo se conserva solo como referencia historica de una comparacion antigua.

El proyecto actual no utiliza ese conjunto de arquitecturas. Las CNN activas y recomendadas para el proyecto son:

- `resnet50`
- `efficientnet_b0`
- `densenet121`

La comparacion vigente debe obtenerse ejecutando la pipeline multiclase actual y consultando los artefactos generados en:

- `Predictor_models/artifacts/metrics`
- `Predictor_models/artifacts/reports/experiment_summary.csv`

Si necesitas un analisis actualizado, conviene regenerarlo despues del nuevo reentrenamiento completo.
