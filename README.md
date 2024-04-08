# Wavelet_CNN_Classifier
![CNN Arquitecture][model_WCNN_arc1a.png]

## Descripción del Código

Este proyecto tiene como objetivo preparar datos, los cuales se extren caracteristicas por  medio de familias wavelet para entrenar un modelo de red neuronal convolucional (CNN) en el contexto de reconocimiento de imágenes y clasificacion de esta. A continuación se describen los pasos principales:

1. **Carga de Datos**: Se cargan imágenes desde un directorio en el sistema de archivos, cada imagen etiquetada con una categoría específica.

2. **Preprocesamiento de Datos**: Se realizan varias transformaciones a las imágenes cargadas, incluyendo la conversión a escala de grises y la normalización de los datos.

3. **Descomposición Wavelet**: Se aplica la descomposición wavelet a las imágenes para extraer características útiles que se utilizarán como entrada para el modelo CNN.

4. **Definición del Modelo CNN**: Se define la arquitectura de una red neuronal convolucional que se utilizará para entrenar el modelo.

5. **Entrenamiento del Modelo**: Se entrena el modelo utilizando los datos preparados, con el objetivo de aprender a clasificar las imágenes en las diferentes categorías.

6. **Evaluación del Modelo**: Se evalúa el rendimiento del modelo utilizando métricas como la precisión y la matriz de confusión.
Este código proporciona una base sólida para trabajar con datos de imágenes y entrenar modelos de aprendizaje automático para tareas de clasificación.
