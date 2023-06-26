# LIBRERIAS
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# CSV
usted_esta_aqui = os.getcwd()
origen = os.path.join(usted_esta_aqui)
os.chdir(origen)
ruta = os.path.join('data', 'raw', 'DF.csv')
df = pd.read_csv(ruta)


# ELIMINAR COLUMNAS QUE NO SE VAN A USAR
df = df.drop(['TITULO', 'GUION'], axis=1)

# ELIMINAR OUTLIERS
df = df[df['DURACION (min)'] <= 200]
df = df[df['VOTOS'] <= 125000]

# FEATURE ENGINEERING

# Nota media por variable
columnas = ['AÑO', 'DURACION (min)', 'GENERO', 'DIRECCION', 'PROTAGONISTA', 'PAIS', 'VOTOS']
for columna in columnas:
        media_por_valor = df.groupby(columna)['NOTA'].mean().round(2)
        nueva_columna = 'NOTA_' + columna
        df[nueva_columna] = df[columna].map(media_por_valor)

# Clasificación por nota media
bins = [6.34, 6.405, 6.47, 6.535, 6.6, 6.665, 6.73, 6.795, 6.86, 6.925, 7]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
df['AÑO_RANKING'] = pd.cut(df['NOTA_AÑO'], bins=bins, labels=labels, right=False).astype(int)
bins = [6.2, 6.375, 6.55, 6.725, 6.9, 7.075, 7.25, 7.425, 7.6, 7.775, 8]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
df['DURACION_RANKING'] = pd.cut(df['NOTA_DURACION (min)'], bins=bins, labels=labels, right=False).astype(int)
bins = [5.97, 6.048, 6.126, 6.204, 6.282, 6.36, 6.438, 6.516, 6.594, 6.672, 7]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
df['GENERO_RANKING'] = pd.cut(df['NOTA_GENERO'], bins=bins, labels=labels, right=False).astype(int)
bins = [5.8, 6.03, 6.26, 6.49, 6.72, 6.95, 7.18, 7.41, 7.64, 7.87, 9]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
df['DIRECCION_RANKING'] = pd.cut(df['NOTA_DIRECCION'], bins=bins, labels=labels, right=False).astype(int)
bins = [5.8, 6.06, 6.32, 6.58, 6.84, 7.1, 7.36, 7.62, 7.88, 8.14, 9]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
df['PROTAGONISTA_RANKING'] = pd.cut(df['NOTA_PROTAGONISTA'], bins=bins, labels=labels, right=False).astype(int)
bins = [5.9, 6.084, 6.268, 6.452, 6.636, 6.82, 7.004, 7.188, 7.372, 7.556, 8]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
df['PAIS_RANKING'] = pd.cut(df['NOTA_PAIS'], bins=bins, labels=labels, right=False).astype(int)
bins = [5.8, 6.07, 6.34, 6.61, 6.88, 7.15, 7.42, 7.69, 7.96, 8.23, 9]
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
df['VOTOS_RANKING'] = pd.cut(df['NOTA_VOTOS'], bins=bins, labels=labels, right=False).astype(int)

# ELIMINAR VARIABLES INICIALES
df = df.drop(['AÑO', 'DURACION (min)', 'GENERO', 'DIRECCION', 'PROTAGONISTA', 'PAIS', 'VOTOS'], axis=1)

# GUARDAR CSV PROCESADO
destino = os.path.join('data', 'processed', 'data.csv')
df.to_csv(destino, index=False)