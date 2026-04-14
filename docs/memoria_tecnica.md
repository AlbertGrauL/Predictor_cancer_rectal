# Memoria Tecnica

## 1. Introduccion

Este proyecto desarrolla un sistema academico multimodal para el estudio de lesiones y patologias relacionadas con cancer colorrectal. El sistema combina dos fuentes de informacion:

- imagenes endoscopicas
- datos tabulares de paciente

La salida por imagen resuelve una clasificacion multiclase y la salida tabular aporta una estimacion binaria de riesgo clinico.

## 2. Objetivo del proyecto

El objetivo no es construir una herramienta asistencial, sino una plataforma de aprendizaje y experimentacion que permita:

- preparar datasets heterogeneos
- entrenar modelos de imagen y tabulares
- comparar resultados con criterios reproducibles
- interpretar predicciones
- presentar los resultados en una interfaz clara

## 3. Alcance funcional actual

La version actual incluye:

- pipeline completo de imagen multiclase
- pipeline completo tabular binario
- comparacion separada de modelos de imagen y tabulares
- evaluacion con curvas, metricas, calibracion y analisis de errores
- integracion en Streamlit con modo combinado

No incluye:

- validacion clinica externa real
- alineacion paciente-imagen a nivel hospitalario
- meta-modelo de fusion entrenado
- uso asistencial

## 4. Modalidad de imagen

### 4.1 Definicion del problema

La tarea visual es clasificar cada imagen endoscopica en una de estas clases:

- `polipo`
- `sano`
- `otras_patologias`

### 4.2 Dataset visual

Fuentes actuales:

- `Casos_negativos/*`
- `Polipos/polyps`
- `Polipos/imagenes con polipos destacados/output/original`
- `Sangre_Paredes/*`

### 4.3 Modelos comparados

- `ResNet50`
- `EfficientNet-B0`
- `DenseNet121`

### 4.4 Entrenamiento visual

Se utiliza transferencia de aprendizaje con:

- congelacion inicial del backbone
- entrenamiento de la cabeza clasificadora
- fine-tuning posterior
- augmentacion moderada
- mascara fija en la esquina inferior izquierda para reducir atajos visuales

### 4.5 Evaluacion visual

La evaluacion de imagen genera:

- accuracy
- precision y recall macro
- F1 macro y weighted
- ROC-AUC y PR-AUC
- matriz de confusion
- metricas por clase
- Grad-CAM en inferencia

## 5. Modalidad tabular

### 5.1 Definicion del problema

La tarea tabular consiste en predecir la variable `cancer_diagnosis` como:

- `sin_riesgo_clinico`
- `riesgo_clinico`

Esta salida no diferencia entre `polipo` y `otras_patologias`; solo aporta una señal binaria de riesgo.

### 5.2 Dataset tabular

Archivo principal:

- `Predictor_models/data/cancer_final.csv`

Formulario base para la app:

- `Predictor_models/data/Preguntas Oncológicas para Pacientes - Preguntas Oncológicas para Pacientes.csv`

### 5.3 Variables usadas

Variables de entrada:

- `age`
- `sex`
- `sof`
- `alcohol`
- `tobacco`
- `diabetes`
- `tenesmus`
- `previous_rt`
- `rectorrhagia`
- `intestinal_habit`
- `digestive_family_history`

Variables excluidas:

- `id`
- `cancer_diagnosis` como entrada del formulario

### 5.4 Preparacion de datos tabulares

La preparacion incluye:

- lectura del CSV con separador `;` y encoding `latin1`
- limpieza de nombres de columnas
- mapeo del objetivo `yes/no -> 1/0`
- codificacion binaria de sintomas y antecedentes
- codificacion numerica de sexo
- mantenimiento de variables ordinales como enteros
- agrupacion de `digestive_family_history` en:
  - `no`
  - `colon`
  - `gastric`
  - `other_positive`
  - `unknown_dirty`
- one-hot encoding de los grupos de antecedentes familiares
- split estratificado `train / val / test`

### 5.5 Modelos tabulares comparados

- `RandomForestClassifier`
- `XGBClassifier`

## 6. Mejora implementada en esta iteracion tabular

Esta iteracion amplía el pipeline tabular con cuatro mejoras principales.

### 6.1 Validacion cruzada

Se aplica validacion cruzada estratificada sobre el split de entrenamiento para estimar mejor el rendimiento y reducir dependencia de una sola particion.

### 6.2 Busqueda de hiperparametros

Se usa `RandomizedSearchCV` para explorar configuraciones razonables en:

- `RandomForest`
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
- `XGBoost`
  - `n_estimators`
  - `max_depth`
  - `learning_rate`
  - `subsample`
  - `colsample_bytree`
  - `reg_lambda`
  - `scale_pos_weight`

### 6.3 Calibracion de probabilidades

Tras elegir el mejor modelo por validacion cruzada, se calibra su salida usando el split de validacion. Esto mejora la interpretacion del riesgo clinico y hace mas fiable la probabilidad mostrada en la app.

### 6.4 Importancia de variables mas robusta

La evaluacion tabular ahora guarda:

- importancia nativa del modelo
- importancia por permutacion

La importancia por permutacion es especialmente util porque mide cuanto empeora el modelo si se desordena cada variable.

## 7. Evaluacion

### 7.1 Imagen

Metricas principales:

- accuracy
- recall macro
- F1 macro
- ROC-AUC
- PR-AUC
- analisis por clase

### 7.2 Tabular

Metricas principales:

- accuracy
- precision positiva
- recall positivo
- F1 positivo
- ROC-AUC
- PR-AUC
- calibracion
- CV best score

### 7.3 Reportes y artefactos

El proyecto guarda metadatos y evaluaciones para ambas modalidades. En tabular se almacenan ademas:

- mejores parametros
- score CV
- resumen de las mejores configuraciones probadas
- importancia por permutacion

## 8. Integracion multimodal en la app

La app tiene tres modos:

- `Solo imagen`
- `Solo datos tabulares`
- `Combinado`

### 8.1 Regla de integracion

La fusion en esta fase no es aprendida. Se basa en una regla explicable:

- la imagen decide la clase final
- el tabular aporta apoyo o alerta de riesgo

### 8.2 Interpretacion combinada

Casos principales:

- imagen `sano` + riesgo tabular alto -> alerta de revision
- imagen `polipo` + riesgo tabular alto -> concordancia alta
- imagen `otras_patologias` + riesgo tabular alto -> concordancia alta
- imagen patologica + riesgo tabular bajo -> no se rebaja la clase, pero se marca baja concordancia

## 9. Limitaciones

- la modalidad tabular no distingue tipo de patologia
- la fusion multimodal aun no usa relacion paciente-imagen real
- la calidad del objetivo tabular limita el techo del modelo
- los resultados siguen dependiendo del contexto del dataset y de la definicion de etiquetas

## 10. Trabajo futuro

Posibles lineas siguientes:

- añadir `CatBoost` o `LightGBM`
- probar fusion entrenada cuando exista correspondencia paciente-imagen fiable
- refinar las categorias de antecedentes familiares
- estudiar umbrales optimizados para sensibilidad clinica
- ampliar la validacion externa

## 11. Aviso final

Este sistema es academico y de investigacion. No sustituye valoracion medica ni debe utilizarse para decisiones clinicas reales.
