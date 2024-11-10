#!/usr/bin/env python
# coding: utf-8
# In[ ]:
#----------------------------------------------------------
# VERSIÓN 2.0 CODIGO CON VARIABLES PREDICTORAS
# SSS
# 8/11/2024
#----------------------------------------------------------
#----------------------------------------------------------
# VERSIÓN 3.0 CODIGO CON DIVISIÓN DE DATOS EN GRUPO TRAIN-TEST
# SSS
# 10/11/2024
#----------------------------------------------------------

import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
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
df_total = df_total.loc[:, ~df_total.columns.duplicated()]

#---------------------------------------------------------------
# Preprocesamiento para normalización y estandarización
#---------------------------------------------------------------

standard_scaler = StandardScaler()
scaler = MinMaxScaler()
df_normalized = df_total.copy()


#---------------------------------------------------------------
# Selección de variables importantes sin duplicados
#---------------------------------------------------------------

variables_importantes_dos = ['AL_SO_PPM', 'AL_SO_SOLIDO_PPM', 'PAC_PPM', 'FECL3_PPM']

variables_importantes_captación = list(set([
    'TEMPERATURA_C_BN', 'OXIGENO_DISUELTO_MG_L_O2_BN', 'TURBIEDAD_UNT_BN',
    'COLOR_UPC_BN', 'CONDUCTIVIDAD_US_CM_BN', 'PH_BN', 'MATERIA_ORGANICA_MG_L_BN',
    'NITROGENO_AMONIACAL_G_L_BN', 'MANGANESOS_MG_L_MN_BN', 'ALCALINIDAD_TOTAL_MG_L_CACO3_BN',
    'CLORUROS_MG_L_CL_BN', 'DUREZA_TOTAL_MG_L_CACO3_BN', 'DUREZA_CALCICA_MG_L_CACO3_BN',
    'HIERRO_TOTAL_MG_L_FE_3_BN', 'ALUMINIO_RESIDUAL_MG_L_AL_BN', 'POTENCIAL_REDOX_MV_BN',
    'NITRITOS_MG_L_NO2_BN', 'FOSFATOS_MG_L_BN', 'NITRATOS_MG_L_NO3_BN', 'SULFATOS_MG_L_SO4_BN', 'COT_MG_L_BN'
]))

variables_importantes_cruda = list(set([
    'TEMPERATURA_C_CRU', 'OXIGENO_DISUELTO_MG_L_O2_CRU', 'TURBIEDAD_UNT_CRU', 
    'COLOR_UPC_CRU', 'CONDUCTIVIDAD_US_CM_CRU', 'PH_CRU', 'MATERIA_ORGANICA_MG_L_CRU',
    'NITROGENO_AMONIACAL_G_L_CRU', 'MANGANESOS_MG_L_MN_CRU', 'ALCALINIDAD_TOTAL_MG_L_CACO3_CRU',
    'CLORUROS_MG_L_CL_CRU', 'DUREZA_TOTAL_MG_L_CACO3_CRU', 'DUREZA_CALCICA_MG_L_CACO3_CRU', 
    'HIERRO_TOTAL_MG_L_FE_3_CRU', 'ALUMINIO_RESIDUAL_MG_L_AL_CRU', 'POTENCIAL_REDOX_MV_CRU',
    'NITRITOS_MG_L_NO2_CRU', 'NITRATOS_MG_L_NO3_CRU', 'FOSFATOS_MG_L_CRU', 'SULFATOS_MG_L_SO4_CRU', 
    'COT_MG_L_CRU', 'SOLIDOS_SUSPENDIDOS_MG_L_CRU'
]))

variables_importantes_mez = list(set([
    'OXIGENO_DISUELTO_MG_L_O2_MEZ', 'TEMPERATURA_C_MEZ', 'PH_MEZ', 
    'CLORO_LIBRE_MG_L_CL2_MEZ', 'CLORO_COMBINADO_MG_L_CL2_MEZ', 'CLORO_TOTAL_MG_L_CL2_MEZ'
]))

# Crear una lista única de todas las variables
todas_las_variables = list(set(variables_importantes_captación + variables_importantes_cruda + variables_importantes_mez))

#---------------------------------------------------------------
# Estandarización de las variables
#---------------------------------------------------------------

scaler = StandardScaler()
df_standardized_global = df_total.copy()
df_standardized_global[todas_las_variables] = scaler.fit_transform(df_total[todas_las_variables])

# Definir X e y después de la estandarización
X = df_standardized_global[todas_las_variables]
df_standardized_global['DOSIS_TOTAL'] = (
    df_standardized_global['PAC_PPM'] + df_standardized_global['AL_SO_PPM'] + 
    df_standardized_global['AL_SO_SOLIDO_PPM'] + df_standardized_global['FECL3_PPM']
)
y = df_standardized_global['DOSIS_TOTAL']

#---------------------------------------------------------------
# Clustering con K-Means y división de datos en Train-Test-Validation
#---------------------------------------------------------------

kmeans = KMeans(n_clusters=2, random_state=42)
clust_labels = kmeans.fit_predict(X)

X_cluster_0 = X[clust_labels == 0]
y_cluster_0 = y[clust_labels == 0]
X_cluster_1 = X[clust_labels == 1]
y_cluster_1 = y[clust_labels == 1]

#---------------------------------------------------------------
# División de datos en Test y Train
#---------------------------------------------------------------

# Para el Cluster 0
X_train_full_0, X_test_0, y_train_full_0, y_test_0 = train_test_split(X_cluster_0, y_cluster_0, test_size=0.2, random_state=42)
X_train_0, X_val_0, y_train_0, y_val_0 = train_test_split(X_train_full_0, y_train_full_0, test_size=0.25, random_state=42)

# Para el Cluster 1
X_train_full_1, X_test_1, y_train_full_1, y_test_1 = train_test_split(X_cluster_1, y_cluster_1, test_size=0.2, random_state=42)
X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(X_train_full_1, y_train_full_1, test_size=0.25, random_state=42)

#print("Cluster 0:")
#print(f"Tamaño de X_train_0: {X_train_0.shape}")
#print(f"Tamaño de X_val_0: {X_val_0.shape}")
#print(f"Tamaño de X_test_0: {X_test_0.shape}")

#print("\nCluster 1:")
#print(f"Tamaño de X_train_1: {X_train_1.shape}")
#print(f"Tamaño de X_val_1: {X_val_1.shape}")
#print(f"Tamaño de X_test_1: {X_test_1.shape}")


#---------------------------------------------------------------
# Entrenamiento Modelos de Regresión
#---------------------------------------------------------------

experiment = mlflow.set_experiment("Regresion-DosisOptima")

with mlflow.start_run(experiment_id=experiment.experiment_id):
    n_estimators_0=250
    n_estimators_1=800
    max_depth_0=36
    max_depth_1=48

    model_0 = RandomForestRegressor(n_estimators=n_estimators_0, max_depth=max_depth_0, random_state=42)
    # model_0.fit(X_cluster_0, y_cluster_0)
    model_0.fit(X_train_0, y_train_0)
    
    model_1 = RandomForestRegressor(n_estimators=n_estimators_1, max_depth=max_depth_1, random_state=42)
    # model_1.fit(X_cluster_1, y_cluster_1)
    model_1.fit(X_train_1, y_train_1)


    #y_pred_0 = model_0.predict(X_cluster_0)
    #y_pred_1 = model_1.predict(X_cluster_1)

    y_pred_0 = model_0.predict(X_val_0)
    y_pred_1 = model_1.predict(X_val_1)

    mse_0 = mean_squared_error(y_val_0, y_pred_0)
    mae_0 = mean_absolute_error(y_val_0, y_pred_0)
    r2_0 = r2_score(y_val_0, y_pred_0)

    mse_1 = mean_squared_error(y_val_1, y_pred_1)
    mae_1 = mean_absolute_error(y_val_1, y_pred_1)
    r2_1 = r2_score(y_val_1, y_pred_1)

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

