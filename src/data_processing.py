# IMPORTAR LIBRERÍAS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# LEER CSV
df = pd.read_csv('data/raw/DF.csv')

# LIMPIAR DATOS

# Borrar columnas inservibles
df = df.drop(['TITULO', 'GUION'], axis=1)

# Outliers
df = df[df['DURACION (min)'] <= 200]

# FEATURE ENGINEERING

# Agrupar años en décadas
bins = [1972, 1980, 1990, 2000, 2010, 2020, 2024]
labels = [70, 80, 90, 100, 110, 120]
df['DECADA'] = pd.cut(df['AÑO'], bins=bins, labels=labels, right=False).astype(int)

# Agrupar minutos en horas
bins = [90, 120, 180, 201]
labels = [1, 2, 3]
df['DURACION (h)'] = pd.cut(df['DURACION (min)'], bins=bins, labels=labels, right=False).astype(int)

# Agrupar votos (popularidad)
bins = [1000, 41000, 82000, 123000, 164000, 225000]
labels = [1, 2, 3, 4, 5]
df['POPULARIDAD'] = pd.cut(df['VOTOS'], bins=bins, labels=labels, right=False).astype(int)

# Codificación de variables categóricas
label_encoder = LabelEncoder()
# Género
label_encoder.fit(df['GENERO'].unique())
df['GENERO_NUM'] = label_encoder.transform(df['GENERO'])
# Dirección
label_encoder.fit(df['DIRECCION'].unique())
df['DIRECCION_NUM'] = label_encoder.transform(df['DIRECCION'])
# Protagonista
label_encoder.fit(df['PROTAGONISTA'].unique())
df['PROTAGONISTA_NUM'] = label_encoder.transform(df['PROTAGONISTA'])
# País
label_encoder.fit(df['PAIS'].unique())
df['PAIS_NUM'] = label_encoder.transform(df['PAIS'])

# Juan Palomo (Yo la dirijo, yo la protagonizo)
df['JUAN_PALOMO'] = (df['DIRECCION'] == df['PROTAGONISTA']).astype(int)

# Medias con respecto la Nota
# Genero
df['GENERO_NM'] = df['GENERO'].map(df.groupby('GENERO')['NOTA'].mean())
# Año
df['AÑO_NM'] = df['AÑO'].map(df.groupby('AÑO')['NOTA'].mean())
# Duración
df['DURACION_NM'] = df['DURACION (h)'].map(df.groupby('DURACION (h)')['NOTA'].mean())
# País
df['PAIS_NM'] = df['PAIS'].map(df.groupby('PAIS')['NOTA'].mean())
# Popularidad
df['POPULARIDAD_NM'] = df['POPULARIDAD'].map(df.groupby('POPULARIDAD')['NOTA'].mean())

# Renombrar columnas
df = df.rename(columns={'DURACION (min)': 'MINUTOS'})
df = df.rename(columns={'DURACION (h)': 'HORAS'})

# Reordenar columnas
new_column_order = ['NOTA', 'AÑO', 'DECADA', 'AÑO_NM', 'MINUTOS', 'HORAS', 'DURACION_NM', 'GENERO', 'GENERO_NUM', 'GENERO_NM', 'DIRECCION', 'DIRECCION_NUM', 'PROTAGONISTA', 'PROTAGONISTA_NUM', 'JUAN_PALOMO', 'PAIS', 'PAIS_NUM', 'PAIS_NM', 'VOTOS', 'POPULARIDAD', 'POPULARIDAD_NM']
df = df[new_column_order]

# GUARDAR CSV EN data/processed
df.to_csv('data/processed/data.csv', index=False)