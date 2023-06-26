# LIBRERIAS
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import joblib
import yaml

# CSV
usted_esta_aqui = os.getcwd()
origen = os.path.join(usted_esta_aqui)
os.chdir(origen)
ruta = os.path.join('data', 'processed', 'data.csv')
df = pd.read_csv(ruta)

# TRAIN / TEST
X = df.drop('NOTA', axis=1)
y = df['NOTA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_destino = os.path.join('data', 'train', 'train.csv')
test_destino = os.path.join('data', 'test', 'test.csv')

train_data.to_csv(train_destino, index=False)
test_data.to_csv(test_destino, index=False)

# ENTRENAR MODELOS

linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

ridge = Ridge()
ridge.fit(X_train, y_train)

lasso = Lasso()
lasso.fit(X_train, y_train)

svr = SVR()
svr.fit(X_train, y_train)

random_forest = RandomForestRegressor()
random_forest.fit(X_train, y_train)

kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(X_train)

# GUARDAR MODELOS EN pkl

linear_regression_destino = os.path.join('models', 'linear_regression.pkl')
ridge_destino = os.path.join('models', 'ridge.pkl')
lasso_destino = os.path.join('models', 'lasso.pkl')
svr_destino = os.path.join('models', 'svr.pkl')
random_forest_destino = os.path.join('models', 'random_forest.pkl')
kmeans_destino = os.path.join('models', 'kmeans.pkl')

joblib.dump(linear_regression, linear_regression_destino)
joblib.dump(ridge, ridge_destino)
joblib.dump(lasso, lasso_destino)
joblib.dump(svr, svr_destino)
joblib.dump(random_forest, random_forest_destino)
joblib.dump(kmeans, kmeans_destino)

# CONFIGURAR MODELOS EN yaml

model_config = {'random_forest': {random_forest},}

yaml_destino = os.path.join('models', 'random_forest.yaml')

with open(yaml_destino, 'w') as config_file:
    yaml.dump(model_config, config_file)