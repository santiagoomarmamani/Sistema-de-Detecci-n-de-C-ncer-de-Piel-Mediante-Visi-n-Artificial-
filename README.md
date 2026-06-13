# Sistema de Detección de Cáncer de Piel mediante Visión Artificial

Proyecto de práctica profesionalizante que utiliza inteligencia artificial para clasificar
imágenes de lesiones cutáneas en 7 categorías, basado en el dataset HAM10000.

## ⚠️ Aviso importante
Este proyecto es una herramienta educativa y de referencia. **No es un dispositivo médico
certificado** y no debe usarse para diagnóstico clínico real. Cualquier lesión sospechosa
debe ser evaluada por un dermatólogo.

## Tecnologías utilizadas
- Python 3.11
- PyTorch (deep learning)
- OpenCV (procesamiento de imágenes)
- Google Colab (entrenamiento con GPU)
- Visual Studio Code

## Dataset
[HAM10000](https://api.isic-archive.com/collections/66/)
— 10,015 imágenes dermatoscópicas clasificadas en 7 tipos de lesión.

## Clases detectadas
1. Melanoma
2. Nevo melanocítico
3. Carcinoma basocelular
4. Queratosis actínica
5. Queratosis benigna
6. Dermatofibroma
7. Lesión vascular

## Estructura del proyecto

skin_cancer_detector/

├── data/          # Dataset (no incluido por tamaño)
├── models/        # Modelos entrenados (no incluido por tamaño)
├── src/
│   ├── preprocess.py   # Transformaciones de imágenes
│   ├── model.py        # Arquitectura del modelo (EfficientNet)
│   ├── train.py        # Script de entrenamiento
│   └── gradcam.py       # Visualización Grad-CAM
└── main.py        # Aplicación principal (cámara/imagen)

## Estado del proyecto
🚧 En desarrollo — Accuracy de validación actual: ~48%

## Bitácora de desarrollo
Ver carpeta `docs/bitacora.md` para el registro detallado de horas y actividades.
