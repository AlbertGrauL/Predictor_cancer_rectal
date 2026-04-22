# # 🏥 Especificación Técnica: Sistema de Soporte a la Decisión Clínica (DSS)
## Proyecto: Predictor de Neoplasias Colorrectales (v1: Ensemble de Expertos)

Este documento constituye la especificación de alto nivel del sistema de diagnóstico asistido por computadora (CADe/CADx) diseñado para la detección precoz de lesiones neoplásicas en mucosa colorrectal. El sistema se fundamenta en una arquitectura de ensemble binario y un pipeline riguroso de limpieza de datos mediante redes generativas.

---

## 1. Arquitectura del Sistema (v1): Ensamble de Especialistas

A diferencia de las arquitectituras multiclase convencionales, el sistema **v1** implementa un **Ensamble de Clasificadores Binarios (One-vs-Rest)**. Esta decisión de diseño se basa en la necesidad clínica de maximizar la **Sensibilidad (Recall)** para patologías críticas sin comprometer la especificidad global.

### Hallazgos Clínicos Identificados:
1.  **Detección de Tejido Polipoideo**: Identificación de crecimientos anómalos (adenomas superficiales, pólipos sésiles y pedunculados). El sistema está entrenado para reconocer morfologías compatibles con la **Clasificación de París** (0-Is, 0-Ip, 0-IIa).
2.  **Identificación de Sangrado Activo**: Detección de hemorragias intraluminales, indicador frecuente de lesiones avanzadas o enfermedad inflamatoria intestinal (EII) erosiva.
3.  **Monitoreo de Inflamación Mucosa**: Reconocimiento de eritemas, pérdida de patrón vascular y friabilidad, signos clave en la evaluación de actividad en colitis ulcerosa o enfermedad de Crohn.
4.  **Confirmación de Mucosa Sana (Negativos)**: Validación de ausencia de patología macroscópica, proporcionando un control de calidad para el flujo de trabajo del endoscopista.

### Motor de Inferencia:
El sistema utiliza un Backbone de **EfficientNet-B0**, seleccionado por su equilibrio óptimo entre profundidad de campo receptivo y latencia de inferencia en tiempo real. La activación final mediante **Sigmoide** permite la detección de múltiples hallazgos concurrentes en un mismo frame (ej. un pólipo con sangrado activo).

---

## 2. Protocolo de Limpieza de Datos Digitales (Inpainting AOT-GAN)

Uno de los principales desafíos en el entrenamiento de IA médica es el sesgo introducido por la telemetría del equipo (texto en pantalla, logos del hospital). El sistema **v1** mitiga este sesgo mediante un pipeline de **Virtualización Mucosa**:

1.  **Detección de Telemetría (Masking)**: Mediante el script `preprocess_masks.py`, el sistema identifica las regiones de texto (coordenadas de metadatos clínicos).
2.  **Inpainting con AOT-GAN**: El script `preprocess_inpaint.py` emplea una red **AOT-GAN (Aggregated Contextual Transformations)**. Esta red reconstruye el tejido oculto tras el texto utilizando texturas de mucosa circundante, eliminando artefactos que de otro modo podrían actuar como "atajos" aprendidos (shortcuts) por la red neuronal.

---

## 3. Especificación del Corpus de Datos

El dataset consolidado representa uno de los bancos de imágenes de endoscopía más robustos disponibles internamente:

- **Total de Imágenes**: 71,159 capturas.
- **Pólipos (Ground Truth)**: 1,798 pares de imagen/máscara perfectamente sincronizados (fuentes: CVC-ClinicDB, Kvasir-SEG y capturas propias).
- **Control Negativo**: 1,500 imágenes de mucosa normal (Ciego, Píloro, Flexuras).
- **Datos Crudos para GAN**: 61,000+ imágenes sin clasificar utilizadas para entrenar la capacidad de reconstrucción de texturas (inpainting).

---

## 4. Análisis de Rendimiento y Telemetría (MLflow)

El rendimiento se monitoriza mediante **MLflow**, priorizando métricas con relevancia clínica:

| Métrica | Importancia Clínica | Objetivo |
| :--- | :--- | :--- |
| **Sensibilidad (Sensitivity)** | Minimización de falsos negativos (lesiones omitidas). | > 92% |
| **Puntaje F1 (F1-Score)** | Balance entre precisión diagnóstica y seguridad clínica. | Optimización Central |
| **AUC-ROC** | Capacidad de discriminación global del modelo. | > 0.95 |

---

## 5. ⚠️ Anexo: Limitaciones y Factores de Interferencia Clínica

Para una interpretación correcta de los resultados, el facultativo debe considerar los siguientes factores de interferencia (Out-of-Distribution cases):

- **Detritos y Residuos (Heces)**: Pueden ser confundidos con lesiones de tipo sésil o inflamación si la preparación colónica es deficiente (Escala de Boston < 6).
- **Artefactos de Iluminación**: Reflejos especulares excesivos en la mucosa húmeda pueden generar falsos positivos en la detección de sangrado.
- **Interferencias Físicas**: Burbujas de aire (biotina) o movimientos bruscos de la cámara que inducen desenfoque (motion blur).
- **Ocultamiento por Pliegues**: Lesiones situadas tras pliegues haustrales pueden presentar un desafío para el modelo si no se obtiene un ángulo de visión completo.

---

> [!NOTE]
> Todo el historial de experimentos, curvas de pérdida y checkpoints de los mejores modelos está disponible en la base de datos central `mlflow.db`.
