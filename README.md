# Recomendador de Películas

Este proyecto es un sistema de recomendación de películas basado en algoritmos de aprendizaje automático (machine learning). 
Proporciona recomendaciones personalizadas de películas a los usuarios utilizando técnicas de regresión y clustering.

## Características
- Utiliza un conjunto de datos de películas que incluye información como año, duración, género, director, protagonista, etc.
- Implementa varios modelos de aprendizaje automático supervisados, como regresión lineal, Ridge, Lasso, SVM y Random Forest, para predecir las calificaciones de las películas.
- Aplica el algoritmo K-means para realizar agrupamiento de películas y generar recomendaciones basadas en la similitud de características.
- Incluye una aplicación web basada en Streamlit para interactuar con el sistema de recomendación y visualizar los resultados.

##Estructura de carpetas
- data: Contiene los conjuntos de datos utilizados para entrenar y evaluar los modelos.
- models: Almacena los modelos entrenados y los archivos relacionados.
- src: Contiene el código fuente del proyecto, incluyendo los archivos de entrenamiento de modelos y evaluación.
- app: Contiene los archivos necesarios para la aplicación web basada en Streamlit.
