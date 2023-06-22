import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('data/processed/data.csv')

X = df.drop('NOTA', axis=1)
y = df['NOTA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('data/train/train.csv', index=False)
test_data.to_csv('data/test/test.csv', index=False)

train_data = pd.read_csv('data/train/train.csv')

# 1. Regresión lineal

numeric_columns = train_data.select_dtypes(include='number')

X = numeric_columns.drop('NOTA', axis=1)  
y = numeric_columns['NOTA']

regression_model = LinearRegression()
regression_model.fit(X, y)

joblib.dump(regression_model, 'models/regresion_lineal.pkl')
