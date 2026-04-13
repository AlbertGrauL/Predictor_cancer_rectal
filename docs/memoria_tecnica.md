# Memoria Tecnica

## 1. Introduccion

Este proyecto desarrolla un sistema academico de clasificacion multiclase de imagenes endoscopicas orientado a la deteccion de polipos, casos sanos y otras patologias relacionadas con el cancer colorrectal.

## 2. Problema y motivacion

Los polipos colorrectales pueden representar lesiones precursoras del cancer colorrectal. En entornos docentes, un sistema de vision por computador permite estudiar:

- preparacion y auditoria de datasets medicos
- transferencia de aprendizaje con CNN
- evaluacion robusta y analisis de errores
- interpretabilidad de predicciones
- despliegue de prototipos accesibles

## 3. Alcance de la v1

La version actual implementa:

- auditoria automatizada del dataset
- pipeline reproducible para entrenamiento multiclase
- comparacion de CNNs modernas
- evaluacion completa con metricas y graficas
- Grad-CAM para interpretacion visual
- app de demostracion con Streamlit

Queda fuera de esta version:

- validacion clinica formal
- integracion hospitalaria
- certificacion o uso asistencial

## 4. Dataset

### 4.1 Fuentes consideradas

- `Casos_negativos/*`
- `Polipos/polyps`
- `Polipos/imagenes con polipos destacados/output/original`
- `Sangre_Paredes/*`

### 4.2 Riesgos del dataset

- desbalance entre clases y subfuentes
- posible fuga de datos entre colecciones similares
- heterogeneidad de resoluciones y formatos
- sesgo de dominio al mezclar datasets de distinto origen

### 4.3 Decision metodologica

Para la configuracion actual:

- entrenar con tres clases: `polipo`, `sano` y `otras_patologias`
- mantener un split reproducible y estratificado
- comparar varias CNN sobre la misma definicion multiclase

## 5. Metodologia de modelado

### 5.1 Modelos comparados

- `ResNet50`: baseline robusta y facil de justificar academicamente
- `EfficientNet-B0`: opcion principal por su buena relacion coste-rendimiento
- `DenseNet121`: alternativa solida en imagen medica

Estas tres arquitecturas se han seleccionado para evitar comparaciones innecesarias y concentrar el proyecto en modelos equilibrados, interpretables y razonables para el volumen de datos disponible.

### 5.2 Estrategia de entrenamiento

- transferencia de aprendizaje
- entrenamiento inicial de la cabeza clasificadora
- fine-tuning parcial o completo segun resultados de validacion
- augmentaciones moderadas y clinicamente razonables

### 5.3 Seleccion del mejor modelo

No se selecciona por accuracy aislada. El criterio principal es:

- recall de `polipo`
- F1 macro
- ROC-AUC
- PR-AUC
- estabilidad entre corridas
- coherencia visual de Grad-CAM

## 6. Evaluacion

### 6.1 Metricas cuantitativas

- accuracy
- precision
- recall
- F1-score
- ROC-AUC
- PR-AUC
- matriz de confusion
- metricas por clase

### 6.2 Evaluacion cualitativa

- revision de falsos positivos
- revision de falsos negativos
- analisis por fuente del dataset
- estudio visual de mapas Grad-CAM

## 7. Interpretabilidad

Se utiliza `Grad-CAM` para mostrar que regiones de la imagen influyen mas en la prediccion. Esto mejora la comprension del comportamiento del modelo y facilita la discusion de casos acertados y erroneos.

## 8. Interfaz Streamlit

La aplicacion incluye:

- carga de imagen
- prediccion multiclase
- probabilidad estimada
- visualizacion Grad-CAM
- breve explicacion en lenguaje natural
- panel de metricas y comparacion entre modelos multiclase

## 9. Limitaciones

- dataset no clinicamente curado para uso asistencial
- resultados dependientes de la calidad y distribucion de las fuentes
- posible falta de generalizacion a otros hospitales o dispositivos
- interpretabilidad visual util, pero no equivalente a causalidad

## 10. Trabajo futuro

- incorporar validacion cruzada mas amplia
- evaluar arquitecturas adicionales
- integrar segmentacion y clasificacion conjunta
- estudiar calibracion avanzada y umbrales clinicamente orientados

## 11. Aviso final

Este sistema debe entenderse como una herramienta academica de aprendizaje y experimentacion. No sustituye el juicio medico ni debe emplearse para toma de decisiones clinicas reales.
