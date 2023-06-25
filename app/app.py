import streamlit as st
import pandas as pd

def main():
    st.title('RECOMENDADOR DE PELÍCULAS')
    st.markdown('by Xavi Albert')

    st.markdown('---')

    st.sidebar.title('Preguntas')
    question_1 = st.sidebar.checkbox('¿Qué problema o necesidad vamos a resolver?')
    question_2 = st.sidebar.checkbox('¿Qué datos se han utilizado para el entrenamiento?')
    question_3 = st.sidebar.checkbox('¿Qué transformaciones y consideraciones se han realizado?')
    question_4 = st.sidebar.checkbox('¿Qué modelos has entrenado?')
    question_5 = st.sidebar.checkbox('¿Cómo ha sido el proceso de entrenamiento?')
    question_6 = st.sidebar.checkbox('¿Qué resultados nos ofrecen? ¿Qué modelo tiene mejores resultados?')
    question_7 = st.sidebar.checkbox('¿Qué variables tienen de mayor impacto?')
    question_8 = st.sidebar.checkbox('¿Qué conclusiones has obtenido?')

    st.markdown('---')

    if question_1:
        st.header('¿Qué problema o necesidad vamos a resolver?')
        st.markdown('''
        El proyecto tiene como objetivo principal desarrollar un modelo de machine learning capaz de predecir la nota que le pondría un usuario a una película en base a sus caraterísticas.
        ''')

    if question_2:
        st.header('¿Qué datos se han utilizado para el entrenamiento?')
        st.markdown('''
        Un conjunto de calificaciones de más de 5000 películas, obtenido de Filmaffinity.
        ''')

    if question_3:
        st.header('¿Qué transformaciones y consideraciones se han realizado?')
        st.markdown('''
        Se ha optado por clasificar las diversas características en base a su puntuación media, con el fin de agruparlas según la preferencia de los usuarios.
        ''')

    if question_4:
        st.header('¿Qué modelos has entrenado?')
        st.markdown('''
        Se entrenaron varios modelos supervisados, incluyendo regresión lineal, Ridge, Lasso, SVR y Random Forest.
        También se utilizó el algoritmo de clustering K-means como modelo no supervisado.
        ''')

    if question_5:
        st.header('¿Cómo ha sido el proceso de entrenamiento?')
        st.markdown('''
        El proceso de entrenamiento consistió en cargar los datos de entrenamiento, realizar las transformaciones
        necesarias, dividir los datos en conjuntos de entrenamiento y prueba, entrenar cada modelo utilizando los
        datos de entrenamiento y realizar la evaluación utilizando los datos de prueba. Se utilizaron métricas de
        evaluación como el R2, MSE, MAE y MAPE para evaluar el rendimiento de cada modelo.
        ''')

    if question_6:
        st.header('¿Qué resultados nos ofrecen? ¿Qué modelo tiene mejores resultados?')
        st.markdown('''
        Los resultados obtenidos muestran que los modelos de regresión lineal, Ridge, Lasso, SVR y Random Forest
        son capaces de predecir la nota de una película. A continuación se presentan los resultados obtenidos para
        cada modelo:

        - Regresión Lineal:
          - R2: 0.919120
          - MAE: 0.101466
          - MAPE: 0.015604
          - MSE: 0.020687
          - RMSE: 0.143830

        - Ridge:
          - R2: 0.919145
          - MAE: 0.101513
          - MAPE: 0.015611
          - MSE: 0.020681
          - RMSE: 0.143807

        - Lasso:
          - R2: -0.000304
          - MAE: 0.411927
          - MAPE: 0.062375
          - MSE: 0.255851
          - RMSE: 0.505817

        - SVR:
          - R2: 0.930250
          - MAE: 0.092490
          - MAPE: 0.014196
          - MSE: 0.017840
          - RMSE: 0.133567

        - Random Forest:
          - R2: 0.942414
          - MAE: 0.061395
          - MAPE: 0.009421
          - MSE: 0.014729
          - RMSE: 0.121363
        ''')

    if question_7:
        st.header('¿Qué variables tienen de mayor impacto?')
        st.markdown('''
        Las variables que tienen mayor impacto en la predicción de la nota de una película pueden variar según el
        modelo utilizado. En el caso de los modelos de regresión lineal, Ridge y Lasso, se puede analizar el coeficiente
        de cada variable para determinar su impacto. Para el modelo de Random Forest, se pueden utilizar las
        características de importancia para identificar las variables más influyentes en la predicción.
        ''')

    if question_8:
        st.header('¿Qué conclusiones has obtenido?')
        st.markdown('''
        En conclusión, se ha logrado desarrollar un sistema de recomendación de películas utilizando diferentes modelos
        supervisados y no supervisados. Los modelos de regresión lineal, Ridge, Lasso, SVR y Random Forest fueron capaces
        de predecir la nota de una película con cierto grado de precisión, destacando el modelo Random Forest como el
        mejor en términos de métricas de evaluación. Además, se identificaron las variables más importantes para la
        predicción, lo que proporciona información valiosa sobre los aspectos que más influyen en la nota de una película.
        En general, el proyecto brinda una base sólida para la recomendación de películas y ofrece oportunidades para
        futuras mejoras y refinamientos del sistema.
        ''')

if __name__ == '__main__':
    main()
