# Predictor Clínico Front-End (Angular)

Este directorio contiene la interfaz de usuario web del **Predictor de Cáncer Colorrectal**, diseñada específicamente para uso clínico por endoscopistas y personal médico.

## 🛠 Arquitectura Tecnológica
- **Framework:** Angular 17+ (Aplicación SPA)
- **Estilos:** Tailwind CSS (con soporte para tipografía y queries de contenedor)
- **Comunicación:** Servicios HTTP asíncronos enlazados a la API de FastAPI.
- **Gestión de Estado:** RxJS y Signals (Angular 17+).

##  Puesta en Marcha (Desarrollo)

Para iniciar la aplicación en tu entorno local:

1. **Instalar Dependencias:** (Solo la primera vez)
   ```bash
   npm install
   ```
2. **Arrancar el Servidor:**
   ```bash
   npm start
   ```
   La aplicación se abrirá en `http://localhost:4200/`.

##  Integración con la API (IA)
El frontend no ejecuta redes neuronales directamente. Actúa como un cliente ligero que se comunica con el servidor de IA.

1. **Flujo de Usuario:**
   - El médico rellena el formulario de triaje clínico (edad, antecedentes, síntomas).
   - El médico sube una imagen endoscópica capturada.
   - El front-end empaqueta estos datos como `multipart/form-data` y los envía a la ruta `POST /predict` de la API (puerto 8000).

2. **Recepción de Resultados:**
   - El backend devuelve un JSON con el diagnóstico de los 4 modelos (Pólipos, Sangre, Inflamación, Negativos) y el puntaje clínico.
   - El frontend renderiza visualmente estos porcentajes mediante barras de progreso y alertas de color (verde/rojo) para facilitar la toma de decisiones rápidas.

*(Nota: La limpieza de la imagen mediante inpainting ocurre en el backend antes de la clasificación, de forma transparente para el frontend).*
