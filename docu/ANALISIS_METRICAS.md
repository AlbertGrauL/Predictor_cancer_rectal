# Guía de Interpretación de Métricas Clínicas

Este documento sirve como referencia para interpretar los resultados generados por el proyecto en MLflow y evaluar el rendimiento de los modelos en el contexto clínico de la endoscopía digestiva.

## 1. Métricas Primarias de Evaluación

Debido al desbalance natural de las patologías (hay más mucosa sana que pólipos o sangrados), la precisión global (Accuracy) **no es una métrica fiable**. El sistema prioriza las siguientes métricas:

### Sensibilidad (Recall / True Positive Rate)
- **Fórmula:** TP / (TP + FN)
- **Significado Clínico:** Es la capacidad del modelo para detectar la patología cuando realmente está ahí.
- **Importancia:** **CRÍTICA**. En oncología preventiva, un Falso Negativo (omitir un adenoma que se convertirá en cáncer) es mucho más peligroso que un Falso Positivo.
- **Objetivo del Proyecto:** > 92%.

### Especificidad (True Negative Rate)
- **Fórmula:** TN / (TN + FP)
- **Significado Clínico:** Es la capacidad del modelo para confirmar que el paciente está sano cuando realmente lo está.
- **Importancia:** Alta. Una baja especificidad causaría demasiadas "falsas alarmas" (Falsos Positivos), generando fatiga de alarmas en el endoscopista.

### F1-Score
- **Significado Clínico:** Representa la media armónica entre la Precisión (cuántas alarmas fueron reales) y la Sensibilidad. Es el mejor indicador del rendimiento general de un especialista bajo condiciones de desbalance severo de datos.

### AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
- **Significado Clínico:** Mide la capacidad global del modelo para distinguir entre dos clases (Ej. Pólipo vs No-Pólipo) a través de todos los umbrales de probabilidad posibles. Un valor de 0.5 equivale a tirar una moneda, y 1.0 es la perfección diagnóstica.
- **Objetivo del Proyecto:** > 0.95.

---

## 2. Monitorización Continua con MLflow

Todo el entrenamiento del `v1_expert_binary` se monitoriza a través de un servidor local de MLflow (`mlflow.db`).

### ¿Cómo interpretar las gráficas de MLflow?
1. **`val_loss` vs `train_loss`:** 
   - Si la línea de entrenamiento (train) sigue bajando pero la de validación (val) sube, el modelo está sufriendo **Overfitting** (memorizando los pólipos de entrenamiento en lugar de aprender el concepto). El sistema incorpora *Early Stopping* automático para evitar esto.
2. **`val_auc`:** 
   - Debe formar una curva cóncava suave que tienda a 1.0. Las caídas bruscas suelen indicar problemas de ruido en el lote de validación.

## 3. Notas sobre la Categoría "Negativos"
Históricamente, el clasificador de la clase "Negativos" ha mostrado métricas de AUC ligeramente inferiores al resto de especialistas. Esto es esperado metodológicamente: la categoría "Sano" es extremadamente heterogénea visualmente (incluye ciego, píloro, flexuras y distintos grados de limpieza colónica según la escala de Boston). En contraste, categorías como "Sangre" tienen descriptores visuales (rojo intenso) mucho más puros y uniformes para una red neuronal.
