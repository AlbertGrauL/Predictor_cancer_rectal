@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m Predictor_models.pipeline.tabular.run_tabular_model_comparison
) else (
  python -m Predictor_models.pipeline.tabular.run_tabular_model_comparison
)

endlocal
