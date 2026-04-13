@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m Predictor_models.pipeline.run_initial_pipeline --clean
) else (
  python -m Predictor_models.pipeline.run_initial_pipeline --clean
)

endlocal
