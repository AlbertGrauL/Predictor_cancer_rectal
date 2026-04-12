# Memoria Tecnica

## 1. Introduccion

Este proyecto desarrolla un sistema academico de clasificacion de imagenes endoscopicas orientado a la deteccion de polipos y a la futura extension hacia otras patologias relacionadas con el cancer colorrectal. La primera entrega se centra en un escenario binario `polipo vs no_polipo`, porque ofrece una base metodologica mas estable, reproducible y defendible para estudios.

## 2. Problema y motivacion

Los polipos colorrectales pueden representar lesiones precursoras del cancer colorrectal. En entornos docentes, un sistema de vision por computador permite estudiar:

- preparacion y auditoria de datasets medicos
- transferencia de aprendizaje con CNN
- evaluacion robusta y analisis de errores
- interpretabilidad de predicciones
- despliegue de prototipos accesibles

## 3. Alcance de la v1

La v1 implementa:

- auditoria automatizada del dataset
- pipeline reproducible para entrenamiento binario
- comparacion de CNNs modernas
- evaluacion completa con metricas y graficas
- Grad-CAM para interpretacion visual
- app de demostracion con Streamlit

Queda fuera de la v1:

- validacion clinica formal
- entrenamiento multiclase definitivo
- integracion hospitalaria
- certificacion o uso asistencial

## 4. Dataset

### 4.1 Fuentes consideradas

- `Casos_negativos/*`
- `Polipos/polyps`
- `Polipos/imagenes con polipos destacados/output/original`
- `Sangre_Paredes/*` solo para evaluacion externa exploratoria

### 4.2 Riesgos del dataset

- desbalance entre clases y subfuentes
- posible fuga de datos entre colecciones similares
- heterogeneidad de resoluciones y formatos
- sesgo de dominio al mezclar datasets de distinto origen

### 4.3 Decision metodologica

Para la fase inicial:

- entrenar solo con `polipo` y `no_polipo`
- reservar `Sangre_Paredes` para analizar falsas activaciones sobre patologias no objetivo
- dejar preparada la configuracion para futuro escenario multiclase

## 5. Metodologia de modelado

### 5.1 Modelos comparados

- `ResNet50`: baseline robusta y facil de justificar academicamente
- `EfficientNet-B0/B2`: opcion principal por su buena relacion coste-rendimiento
- `DenseNet121`: alternativa solida en imagen medica
- `ResNet18`: baseline ligera para pruebas rapidas
- `ConvNeXt-Tiny`: opcion moderna si el hardware lo permite

### 5.2 Estrategia de entrenamiento

- transferencia de aprendizaje
- entrenamiento inicial de la cabeza clasificadora
- fine-tuning parcial o completo segun resultados de validacion
- augmentaciones moderadas y clinicamente razonables

### 5.3 Seleccion del mejor modelo

No se selecciona por accuracy aislada. El criterio principal es:

- recall de `polipo`
- F1-score
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
- curva de calibracion basica

### 6.2 Evaluacion cualitativa

- revision de falsos positivos
- revision de falsos negativos
- analisis por fuente del dataset
- estudio visual de mapas Grad-CAM

### 6.3 Evaluacion externa exploratoria

El conjunto `Sangre_Paredes` se usa para estimar si el modelo binario confunde otras patologias con polipos. Este analisis ayuda a justificar una futura ampliacion multiclase.

## 7. Interpretabilidad

Se utiliza `Grad-CAM` para mostrar que regiones de la imagen influyen mas en la prediccion. Esto mejora la comprension del comportamiento del modelo y facilita la discusion de casos acertados y erroneos.

## 8. Interfaz Streamlit

La aplicacion incluye:

- carga de imagen
- prediccion binaria
- probabilidad estimada
- visualizacion Grad-CAM
- breve explicacion en lenguaje natural
- panel de metricas del mejor modelo

## 9. Limitaciones

- dataset no clinicamente curado para uso asistencial
- resultados dependientes de la calidad y distribucion de las fuentes
- posible falta de generalizacion a otros hospitales o dispositivos
- interpretabilidad visual util, pero no equivalente a causalidad

## 10. Trabajo futuro

- clasificacion multiclase `polipo / no_polipo / otras_patologias`
- incorporar validacion cruzada mas amplia
- evaluar arquitecturas adicionales
- integrar segmentacion y clasificacion conjunta
- estudiar calibracion avanzada y umbrales clinicamente orientados

## 11. Aviso final

Este sistema debe entenderse como una herramienta academica de aprendizaje y experimentacion. No sustituye el juicio medico ni debe emplearse para toma de decisiones clinicas reales.
