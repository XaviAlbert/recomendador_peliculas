# LIBRERIAS
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, silhouette_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import os

# CSV
usted_esta_aqui = os.getcwd()
origen = os.path.join(usted_esta_aqui)
os.chdir(origen)
ruta = os.path.join('data', 'test', 'test.csv')
df = pd.read_csv(ruta)
test_data = pd.read_csv(ruta)

# CARGAR MODELOS

linear_regression_ruta = os.path.join('models', 'linear_regression.pkl')
ridge_ruta = os.path.join('models', 'ridge.pkl')
lasso_ruta = os.path.join('models', 'lasso.pkl')
svr_ruta = os.path.join('models', 'svr.pkl')
random_forest_ruta = os.path.join('models', 'random_forest.pkl')
kmeans_ruta = os.path.join('models', 'kmeans.pkl')

linear_regression = joblib.load(linear_regression_ruta)
ridge = joblib.load(ridge_ruta)
lasso = joblib.load(lasso_ruta)
svr = joblib.load(svr_ruta)
random_forest = joblib.load(random_forest_ruta)
kmeans = joblib.load(kmeans_ruta)

X_test = test_data.drop('NOTA', axis=1)  
y_test = test_data['NOTA']

# EVALUAR MODELOS SUPERVISADOS

linear_regression_pred = linear_regression.predict(X_test)
ridge_pred = ridge.predict(X_test)
lasso_pred = lasso.predict(X_test)
svr_pred = svr.predict(X_test)
random_forest_pred = random_forest.predict(X_test)

metrics = {
    'linear_regression': [r2_score(y_test, linear_regression_pred),
                          mean_absolute_error(y_test, linear_regression_pred),
                          mean_absolute_percentage_error(y_test, linear_regression_pred),
                          mean_squared_error(y_test, linear_regression_pred),
                          np.sqrt(mean_squared_error(y_test, linear_regression_pred))],
    'ridge': [r2_score(y_test, ridge_pred),
              mean_absolute_error(y_test, ridge_pred),
              mean_absolute_percentage_error(y_test, ridge_pred),
              mean_squared_error(y_test, ridge_pred),
              np.sqrt(mean_squared_error(y_test, ridge_pred))],
    'lasso': [r2_score(y_test, lasso_pred),
              mean_absolute_error(y_test, lasso_pred),
              mean_absolute_percentage_error(y_test, lasso_pred),
              mean_squared_error(y_test, lasso_pred),
              np.sqrt(mean_squared_error(y_test, lasso_pred))],
    'svr': [r2_score(y_test, svr_pred),
            mean_absolute_error(y_test, svr_pred),
            mean_absolute_percentage_error(y_test, svr_pred),
            mean_squared_error(y_test, svr_pred),
            np.sqrt(mean_squared_error(y_test, svr_pred))],
    'random_forest': [r2_score(y_test, random_forest_pred),
                      mean_absolute_error(y_test, random_forest_pred),
                      mean_absolute_percentage_error(y_test, random_forest_pred),
                      mean_squared_error(y_test, random_forest_pred),
                      np.sqrt(mean_squared_error(y_test, random_forest_pred))]
}

results_df = pd.DataFrame(metrics, index=['R2', 'MAE', 'MAPE', 'MSE', 'RMSE'])
print(results_df)

# EVALUAR MODELO NO SUPERVISADO

kmeans_pred = kmeans.predict(X_test)

silueta = silhouette_score(X_test, kmeans_pred)
print('Clustering silhouette:', silueta)