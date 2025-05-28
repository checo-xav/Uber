#"Library"
import os #Directorio de trabajo en mac
import pandas as pd # Para manejo de datos
import numpy as np #  Para manejo de datos
import matplotlib.pyplot as plt # Para graficos
#import streamlit as st #Para la app web
import seaborn as sns  # Para mejorar la estética de los gráficos
#import plotly.express as px #para graficos 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV
#from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor


#"Funciones"
from src import data_processing, Calculated_Tables


#"Directorio Mac"
os.getcwd()
os.chdir('/Users/checo_xav/Documents/Uber')



#"Carga de datos"
def cargar_datos():
    return data_processing.importar_datos()
data = cargar_datos()
#print(data.dtypes)


#2 & 3. Data Cleaning/Preprocessing & Feature Engineering"
#Notas importantes: 
# - territory : # Corrección de territorio, replace('Long Tail - Region', 'Long Tail')
# - driver_uuid: 12,550 NA - 1.26%, #Tratamiento a los missing values como "Desconocido"
# - courier_flow: # Tratamiento de missing values como "Desconocidos".#Entre "Motorbike", "UberEats", "Desconocidos" y  "Logistics" acumulan el 98.6 %. El resto se dropea, no suena muy uber eats. 
# - geo_archetype: Drive momentum, Defend CP, Play offense acumulan el 93.8 %, puedo categorizar el resto como otros    
# - dropoff_distance &  dropoff_distance =  12,847 NA = 1.28%: La info faltante en pickup_distance son las mimsas filas que dropoff_distance,dropear ya que solo representan el 1.28% se van
# - restaurant_offered_timestamp_utc: 165 NA - 0.02%
# - geo_archetype: Entre Drive momentum, Defend CP y Play offense suman el 93%
# - merchant_surface: Entre POS, Tablet, Other acumulan el 87.1
def transformar_datos(data):
    return data_processing.data_transformation(data)
df = transformar_datos(data)
df = df.sort_values(by='date').reset_index(drop=True)


#------------------- EXPLORACION ---------------------
print(df.dtypes)
#Estado de la información
for col in df.columns:
    print(f"\n Columna: {col}")
    print(f"Valores únicos (incluyendo NaN): {df[col].nunique(dropna=False)}")

    # Conteo de valores y porcentajes
    conteo = df[col].value_counts(dropna=False)
    porcentajes = (conteo / len(df)) * 100

    resumen = pd.DataFrame({
        'Conteo': conteo,
        'Porcentaje': porcentajes.round(1)
    })

    print(resumen)

    # Porcentaje de NaN
    conteo_na = df[col].isna().sum()
    porcentaje_na = (conteo_na / len(df)) * 100

    print(f"\n Porcentaje de NA en {col}: {porcentaje_na :.2f}%")
del(col,conteo,porcentajes,resumen,conteo_na,porcentaje_na)
#"Exploracion de ATD"

print(df.dtypes)

# Boxplot geo_archetype:
plt.figure(figsize=(10, 6))
sns.boxplot(x='geo_archetype', y='ATD', data=df)

# Boxplot courier_flow: 
plt.figure(figsize=(10, 6))
sns.boxplot(x='courier_flow', y='ATD', data=df)

# Boxplot hora: 
plt.figure(figsize=(10, 6))
sns.boxplot(x='hora', y='ATD', data=df)

# Boxplot weekday: 
plt.figure(figsize=(10, 6))
sns.boxplot(x='weekday', y='ATD', data=df)

# Boxplot territory:
plt.figure(figsize=(10, 6))
sns.boxplot(x='territory', y='ATD', data=df)
#----------------------------------------


######## MODELO DE REG LINEAL MULTIPLE CON REG LASSO ##########

# Separar target y features
df = df.drop(['workflow_uuid', 'driver_uuid', 'delivery_trip_uuid', 'tiempo_total','restaurant_offered_timestamp_utc', 'order_final_state_timestamp_local','eater_request_timestamp_local', 'restaurant_offered_timestamp_local', 'date'], axis=1)

## Unit Standarization
numeric_features = ['pickup_distance', 'dropoff_distance', 'distancia_total']
categorical_features = ['territory', 'courier_flow', 'geo_archetype', 'merchant_surface', 'weekday', 'hora']

target = 'ATD'
X = df.drop('ATD', axis=1)
y = df['ATD']

# Preprocesador: estandariza numéricas y one-hot a categóricas (quitando la primera)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='drop'
)

# Pipeline LassoCV con alpha optimizado por CV
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LassoCV(cv=5, random_state=42, n_alphas=20))
])

# Split train-test conservando el orden temporal
split_index = int(0.8 * len(df))
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# Ajustar modelo
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# EVALUACION DEL MODELO
#mse
mse = mean_squared_error(y_test, y_pred)

# R^2 y R^2 ajustado
r2 = r2_score(y_test, y_pred)
n = X_test.shape[0]
p = X_test.shape[1]
r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - p - 1)

#Esta feoooooooo
print(mse)
print(r2)
print(r2_ajustado)




######### MODELO DE RANDOM FOREST ##########
# Copia el dataframe original
#df_rf = df.drop(['workflow_uuid', 'driver_uuid', 'delivery_trip_uuid', 'tiempo_total','restaurant_offered_timestamp_utc', 'order_final_state_timestamp_local','eater_request_timestamp_local', 'restaurant_offered_timestamp_local', 'date'], axis=1)
df = transformar_datos(data)
df = df.sort_values(by='date').reset_index(drop=True)  # <- Aseguramos orden temporal
df_rf = df.drop(['workflow_uuid', 'driver_uuid', 'delivery_trip_uuid', 'tiempo_total','restaurant_offered_timestamp_utc', 'order_final_state_timestamp_local','eater_request_timestamp_local', 'restaurant_offered_timestamp_local', 'date', 'week_number'], axis=1)

# Crear variables circulares para hora (mejor manejo para columna hora)
df_rf['hora'] = df_rf['hora'].astype(int)
df_rf['territory'] = df_rf['territory'].astype(str)
df_rf['courier_flow'] = df_rf['courier_flow'].astype(str)
df_rf['geo_archetype'] = df_rf['geo_archetype'].astype(str)
df_rf['merchant_surface'] = df_rf['merchant_surface'].astype(str)
df_rf['hora_sin'] = np.sin(2 * np.pi * df_rf['hora'] / 24)
df_rf['hora_cos'] = np.cos(2 * np.pi * df_rf['hora'] / 24)
df_rf = df_rf.drop('hora', axis=1)

# Separar variables categóricas (objetos)
cat_cols_rf = df_rf.select_dtypes(include=['object', 'category']).columns.tolist()
if 'ATD' in cat_cols_rf:
    cat_cols_rf.remove('ATD')
    
    
# Crear dummies
df_rf = pd.get_dummies(df_rf, columns=cat_cols_rf, drop_first=True)
# Separar target y features

X_rf = df_rf.drop('ATD', axis=1)
y_rf = df_rf['ATD']

# Split respetando orden temporal
split_index = int(0.8 * len(df_rf))
X_train_rf, X_test_rf = X_rf.iloc[:split_index], X_rf.iloc[split_index:]
y_train_rf, y_test_rf = y_rf.iloc[:split_index], y_rf.iloc[split_index:]

# Crear modelo Random Forest
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    min_samples_leaf=5,
    max_features=0.5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_rf, y_train_rf)

# Predicción
y_pred_rf = rf.predict(X_test_rf)

# Métricas
mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
r2_rf = r2_score(y_test_rf, y_pred_rf)

print(f"MSE test Random Forest: {mse_rf:.4f}")
print(f"R2 test Random Forest: {r2_rf:.4f}")

# Calcular R2 ajustado
n = X_test_rf.shape[0]
p = X_test_rf.shape[1]

r2_adj_rf = 1 - (1 - r2_rf) * (n - 1) / (n - p - 1)
r2_adj_rf
print(f"R2 ajustado test Random Forest: {r2_adj_rf:.4f}")

# Importancia de variables
importances = pd.Series(rf.feature_importances_, index=X_rf.columns)
print(importances.sort_values(ascending=False).head(10))

######### MODELO DE HistGradientBoostingRegressor ##########

hgb = HistGradientBoostingRegressor(max_iter=100, max_depth=10, random_state=42)
hgb.fit(X_train_rf, y_train_rf)

y_pred_hgb = hgb.predict(X_test_rf)

print("MSE:", mean_squared_error(y_test_rf, y_pred_hgb))
print("R2:", r2_score(y_test_rf, y_pred_hgb))

