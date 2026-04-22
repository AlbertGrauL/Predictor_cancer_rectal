@echo off
title Predictor Cancer Rectal - Master Launcher
color 0B

:menu
cls
echo ========================================================
echo        PREDICTOR CANCER RECTAL - MENU PRINCIPAL
echo ========================================================
echo.
echo [1] ENTRENAMIENTO (v1: 4 Modelos Expertos)
echo [2] ANALISIS: Abrir Dashboard MLflow
echo [3] APLICACION: Arrancar API + Frontend
echo [4] SALIR
echo.
echo ========================================================
set /p opt="Selecciona una opcion: "

if %opt%==1 goto train
if %opt%==2 goto mlflow
if %opt%==3 goto app
if %opt%==4 exit
goto menu

:train
cls
echo --- Iniciando Entrenamiento Secuencial (v1) ---
echo Se entrenaran: Polipos, Sangre, Inflamacion y Negativos.
python -m Predictor_models.pipeline.v1_expert_binary.run_full_pipeline
pause
goto menu

:mlflow
cls
echo --- Iniciando Dashboard MLflow ---
start "MLflow Dashboard" cmd /c "mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo Dashboard abierto en: http://127.0.0.1:5000
pause
goto menu

:app
cls
echo --- Arrancando Sistema Completo ---

:: Iniciar API en segundo plano
echo [1/2] Iniciando API (FastAPI) en puerto 8000...
start "Predictor API" cmd /k "uvicorn Predictor_api.main:app --reload --port 8000"

:: Iniciar Frontend
echo [2/2] Iniciando Frontend (Angular) en puerto 4200...
cd Predictor_front
start "Predictor Frontend" cmd /k "npm start"
cd ..

echo.
echo El sistema deberia estar listo en unos segundos.
echo API: http://127.0.0.1:8000
echo Front: http://localhost:4200
pause
goto menu
