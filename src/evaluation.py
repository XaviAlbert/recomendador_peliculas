import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

test_data = pd.read_csv('data/test/test.csv')

# 1. Regresión lineal

numeric_columns = test_data.select_dtypes(include='number')

X_test = numeric_columns.drop('NOTA', axis=1) 
y_test = numeric_columns['NOTA']

regression_model = joblib.load('models/regresion_lineal.pkl')

y_pred = regression_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print("Regresión lineal (MSE):", mse)
