#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import mlflow
import mlflow.sklearn

#---------------------------------------------------------------
# Obtención y limpieza de datos
#---------------------------------------------------------------

current_dir = os.getcwd()
file_path = os.path.join(current_dir, "Data", "DATOS2023.xlsx")
file_path1 = os.path.join(current_dir, "Data", "DATOS2024.xlsx")

df_2023 = pd.read_excel(file_path)
df_2024 = pd.read_excel(file_path1)

df_2023 = df_2023.rename(columns={"Unnamed: 0": "DÍA"})
df_2024 = df_2024.rename(columns={"Unnamed: 0": "DÍA"})

df_total = pd.concat([df_2023, df_2024], ignore_index=True)
df_total.columns = df_total.columns.str.strip().str.upper()

df_total.replace('ND', 0, inplace=True)
df_total.replace(['X', 'S'], pd.NA, inplace=True)
df_total['MATERIA ORGANICA MG/L BN'].fillna(df_total['COT  MG/L  BN'])
df_total = df_total.apply(pd.to_numeric, errors='coerce')
df_total.fillna(df_total.mean(), inplace=True)
df_total['FECHA'] = pd.to_datetime(df_total['FECHA'], errors='coerce')
df_total = df_total.loc[:, ~df_total.columns.duplicated()]
df_total.drop(columns=['MES'], inplace=True)
df_total.columns = df_total.columns.str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)

#---------------------------------------------------------------
# Preprocesamiento para normalización y estandarización
#---------------------------------------------------------------

scaler = MinMaxScaler()
df_normalized = df_total.copy()
variables_importantes_dos = ['AL_SO_PPM', 'AL_SO_SOLIDO_PPM', 'PAC_PPM', 'FECL3_PPM']
df_normalized[variables_importantes_dos] = scaler.fit_transform(df_total[variables_importantes_dos])

standard_scaler = StandardScaler()
df_standardized = df_total.copy()
df_standardized[variables_importantes_dos] = standard_scaler.fit_transform(df_total[variables_importantes_dos])

df_standardized['DOSIS_TOTAL'] = (
    df_standardized['PAC_PPM'] + df_standardized['AL_SO_PPM'] + 
    df_standardized['AL_SO_SOLIDO_PPM'] + df_standardized['FECL3_PPM']
)

X = df_standardized[variables_importantes_dos]
y = df_standardized['DOSIS_TOTAL']


#---------------------------------------------------------------
# Clustering con K-Means y Entrenamiento de modelos
#---------------------------------------------------------------

kmeans = KMeans(n_clusters=2, random_state=42)
clust_labels = kmeans.fit_predict(X)

X_cluster_0 = X[clust_labels == 0]
y_cluster_0 = y[clust_labels == 0]

X_cluster_1 = X[clust_labels == 1]
y_cluster_1 = y[clust_labels == 1]


experiment = mlflow.set_experiment("Regresion-DosisOptima")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    n_estimators_0=500
    n_estimators_1=500
    max_depth_0=48
    max_depth_1=48

    model_0 = RandomForestRegressor(n_estimators=n_estimators_0, max_depth=max_depth_0, random_state=42)
    model_0.fit(X_cluster_0, y_cluster_0)
    
    model_1 = RandomForestRegressor(n_estimators=n_estimators_1, max_depth=max_depth_1, random_state=42)
    model_1.fit(X_cluster_1, y_cluster_1)


    y_pred_0 = model_0.predict(X_cluster_0)
    y_pred_1 = model_1.predict(X_cluster_1)

    mse_0 = mean_squared_error(y_cluster_0, y_pred_0)
    mae_0 = mean_absolute_error(y_cluster_0, y_pred_0)
    r2_0 = r2_score(y_cluster_0, y_pred_0)

    mse_1 = mean_squared_error(y_cluster_1, y_pred_1)
    mae_1 = mean_absolute_error(y_cluster_1, y_pred_1)
    r2_1 = r2_score(y_cluster_1, y_pred_1)

    mlflow.log_param("n_estimators_0", n_estimators_0)
    mlflow.log_param("max_depth_0", max_depth_0)  
    mlflow.log_param("n_estimators_1", n_estimators_1)
    mlflow.log_param("max_depth_1", max_depth_1) 
    mlflow.log_metric("MSE_Cluster_0", mse_0)
    mlflow.log_metric("MAE_Cluster_0", mae_0)
    mlflow.log_metric("R2_Cluster_0", r2_0)
    mlflow.log_metric("MSE_Cluster_1", mse_1)
    mlflow.log_metric("MAE_Cluster_1", mae_1)
    mlflow.log_metric("R2_Cluster_1", r2_1)

    mlflow.sklearn.log_model(model_0, "RandomForestRegressor_Cluster_0")
    mlflow.sklearn.log_model(model_1, "RandomForestRegressor_Cluster_1")

    print("Modelos de Regresión por Clúster - Métricas registradas en MLflow")
    print(f"Cluster 0 - MSE: {mse_0}, MAE: {mae_0}, R2 Score: {r2_0}")
    print(f"Cluster 1 - MSE: {mse_1}, MAE: {mae_1}, R2 Score: {r2_1}")

