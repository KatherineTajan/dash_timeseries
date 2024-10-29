##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                                              IMPORTACION DE LIBRERÍAS                                                                                          #
#                                                                                                                                                                                #
##################################################################################################################################################################################


from tensorflow.keras.models import load_model

import seaborn as sns
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import colorlover as cl
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import dash
import math
from dash import Dash, Input, Output, callback, dash_table, html, dcc 
from sqlalchemy import create_engine
import keras

# Load sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from math import *
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.compose import ColumnTransformer


import sklearn.metrics
import plotly.express as px
from sklearn.metrics import auc
from sklearn.metrics import classification_report, confusion_matrix

import base64
import urllib
import json
import csv
import os

# Load libraries
import seaborn as sns
from plotly.subplots import make_subplots


# Load sklearn
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from scipy.stats import mannwhitneyu
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
import itertools
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.arima_model import ARIMA
import pickle
import statsmodels.api as sm
import statsmodels.tsa.api as smtsa
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
import itertools
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


#librerías para Perceptrón Multicapa
import sys
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Dropout
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import layers
from scikeras.wrappers import KerasRegressor


#librerías para LSTM 
from keras.layers import Dense, Input, Dropout
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

#librerías para ANN
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import EarlyStopping


#librerías para XGBOOTS
import xgboost as xgb
from sklearn.metrics import mean_absolute_error


#Omitier warnings

import os
import warnings
warnings.filterwarnings("ignore")


##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                                              TRATAMIENTO DE DATOS                                                                                              #
#                                                                                                                                                                                #
##################################################################################################################################################################################

df_models_otros = pd.read_csv('df_ermita_st.csv') #RUTA DE DIRECTORIO DE TRABAJO DE CONTENEDOR

df_models = pd.read_csv('df_ermita_st.csv') #RUTA DE DIRECTORIO DE TRABAJO DE CONTENEDOR
df_models = df_models.drop(columns=["PM10", "year", "month", "day", "hour", "imputado"])
df_models['fecha'] = pd.to_datetime(df_models['fecha'])
df_models.set_index('fecha', inplace=True)

########################################### Función de división de la data según el horizonte ###########################################
def create_time_series_datasets(df, column_name, tau):
    """
    Crea conjuntos de entrenamiento, validación y prueba para una serie de tiempo.

    Parameters:
        df: Serie de tiempo (pandas DataFrame).
        column_name: Nombre de la columna que contiene los datos.
        tau: Horizonte (número de días).

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test: Matrices correspondientes.
    """
    # Asegurarse de que df sea una Serie
    if isinstance(df, pd.DataFrame):
        df = df[column_name].values  # Convertir a un array de numpy

    n = len(df)

    # Verificar que haya suficientes datos para entrenamiento, validación y prueba
    if n <= (tau * 3):
        raise ValueError("No hay suficientes datos para crear los conjuntos.")

    # Crear matrices de entrenamiento
    M_rows_train = n - (tau * 3)  # Filas para entrenamiento
    X_train, y_train = [], []
    
    # Generar conjunto de entrenamiento
    for i in range(tau, M_rows_train):  # las features dependen de tau
        X_train.append(df[i-tau:i])  # Regressors (tau días)
        y_train.append(df[i])  # Target

    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1, 1)

    # Ajustar las filas para validación y prueba
    val_size = tau  # Tamaño de validación basado en tau
    test_size = tau  # Tamaño de prueba basado en tau

    # Generar conjunto de validación
    X_val, y_val = [], []
    for i in range(M_rows_train, M_rows_train + val_size):
        X_val.append(df[i-tau:i])  # Regressors (tau días)
        y_val.append(df[i])  # Target solo un valor para cada y_val

    X_val = np.array(X_val)
    y_val = np.array(y_val).reshape(-1, 1)

    # Generar conjunto de prueba
    X_test, y_test = [], []
    for i in range(M_rows_train + val_size, M_rows_train + val_size + test_size):
        X_test.append(df[i-tau:i])  # Regressors (tau días)
        y_test.append(df[i])  # Target solo un valor para cada y_test

    X_test = np.array(X_test)
    y_test = np.array(y_test).reshape(-1, 1) 

    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = create_time_series_datasets(df_models, 'PM10_imputado', 24)



################################ DATA OTROS MODELOS ###########################

#Definir el tamaño de los conjuntos: 70% entrenamiento, 15% validación y 15% prueba

y = df_models_otros[['fecha', 'PM10_imputado']]  # Se seleccionan todas las columnas relevantes

# Convertir la columna 'time' a datetime si no lo es y ponerla como índice
y['fecha'] = pd.to_datetime(y['fecha'])
y.set_index('fecha', inplace=True)#

# Definir tamaños para los conjuntos
train_size = int(len(y) * 0.7)  # 70% de los datos para el entrenamiento
val_size = int(24)   # 24 horas
test_size = val_size  # El tamaño de prueba debe coincidir con el de validación

# Dividir los datos en entrenamiento, validación y prueba
train = y.iloc[:train_size].copy()
val = y.iloc[train_size:train_size + val_size].copy()
test = y.iloc[train_size + val_size:train_size + val_size + test_size].copy()

#convertiri a serie de tiempo 

train_ts = pd.Series(data=train['PM10_imputado'].values, index=train.index)
val_ts = pd.Series(data=val['PM10_imputado'].values, index=val.index)
test_ts = pd.Series(data=test['PM10_imputado'].values, index=test.index)

#tamaños de datos de train y test

tau_val = len(val_ts)
tau_test = len(test_ts)

# Verificar los tamaños
print(f"Tamaño del conjunto de entrenamiento: {len(train)}")
print(f"Tamaño del conjunto de validación: {len(val)}")
print(f"Tamaño del conjunto de Prueba: {len(test)}")


################################## Función para data de gráficas ############################################


def data_plot(data_rn, horizonte): 

    val_train = len(data_rn) - (horizonte * 2)
    val_val =   val_train + horizonte
    val_test = val_val + horizonte

    # Filtrar los datos según el rango especificado usando iloc
    data_train_plot = data_rn.iloc[0:val_train]  # Filtra de la fila 0 a la 4887
    data_train_plot.index = data_rn.index[0:val_train]  # Asigna el índice correspondiente
    data_train_plot = data_train_plot.squeeze()

    data_val_plot = data_rn.iloc[val_train:val_val]  # Filtra de la fila 4888 a la 4916
    data_val_plot.index = data_rn.index[val_train:val_val]  # Asigna el índice correspondiente
    data_val_plot = data_val_plot.squeeze()


    data_test_plot = data_rn.iloc[val_val:val_test]  # Filtra de la fila 4917 a la 4998
    data_test_plot.index = data_rn.index[val_val:val_test]  # Asigna el índice correspondiente
    data_test_plot = data_test_plot.squeeze()

    # Verifica los resultados
    print("Datos de entrenamiento:")
    print(f"{data_train_plot}{horizonte}")
    print("\nDatos de validación:")
    print(f"{data_val_plot}{horizonte}")
    print("\nDatos de prueba:")
    print(f"{data_test_plot}{horizonte}")

    return data_train_plot, data_val_plot, data_test_plot

data_train_plot, data_val_plot, data_test_plot = data_plot(df_models, 24)



##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                               DEFINICIÓN DE FUNCIONES MODELOS ST                                                                                               #
#                                                                                                                                                                                #
##################################################################################################################################################################################


#Función gráfico de predicciones 

def plot_model(train, val, test, y_pred_val, y_pred_test, title):
    # Crear la figura
    fig = go.Figure()

    # Añadir la serie de entrenamiento
    fig.add_trace(go.Scatter(
        x=train.index, 
        y=train, 
        mode='lines', 
        name='Entrenamiento',
        line=dict(color='blue')
    ))

    # Añadir la serie de validación
    fig.add_trace(go.Scatter(
        x=val.index, 
        y=val, 
        mode='lines', 
        name='Validación',
        line=dict(color='orange')
    ))

    # Añadir la serie de predicción validación
    fig.add_trace(go.Scatter(
        x=val.index,  
        y=y_pred_val, 
        mode='lines', 
        name='Predicción Validación',
        line=dict(color='red', dash='dash')  # Líneas punteadas para las predicciones
    ))

    # Añadir la serie de test
    fig.add_trace(go.Scatter(
        x=test.index, 
        y=test, 
        mode='lines', 
        name='Prueba',
        line=dict(color='green')
    ))

    # Añadir la serie de predicción test
    fig.add_trace(go.Scatter(
        x=test.index, 
        y=y_pred_test, 
        mode='lines', 
        name='Predicción Prueba',
        line=dict(color='purple', dash='dash')  # Líneas punteadas para las predicciones
    ))

    # Calcular el MAE para la validación
    T_val = len(val)
    yh_val = val.copy().values
    prederr_val = yh_val - y_pred_val
    mae_val = round(sum(abs(prederr_val)) / T_val, 2)

    # Calcular el MAE para la prueba
    T_test = len(test)
    yh_test = test.copy().values
    prederr_test = yh_test - y_pred_test
    mae_test = round(sum(abs(prederr_test)) / T_test, 2)

    # Actualizar el diseño de la gráfica
    fig.update_layout(
        title=f"{title}, MAD Validación: {mae_val}, MAD Prueba: {mae_test}",
        xaxis_title="Tiempo",
        yaxis_title="Valor",
        legend_title="Series",
        template="plotly_white",  # Usar fondo blanco
        xaxis=dict(range=[train.index.min(), test.index.max()])  # Ajustar el rango del eje x
    )
    # Mostrar la gráfica interactiva
    return fig


#*********************************************************************** FUNCION ARIMA FORECAST**********************************************************************************************


def ses_optimizer(train, val, alphas, step):

    best_alpha, best_mae = None, float("inf")

    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(val, y_pred)

        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

    return best_alpha, best_mae

def ses_model_tuning(train, val, test, step, title="Model Tuning - Suavización Exponencial Simple"):
    alphas = np.arange(0.8, 1, 0.01)
    best_alpha, best_mae = ses_optimizer(train, val, alphas, step=step)
    
    train_val = pd.concat([train, val])
    final_model = SimpleExpSmoothing(train_val).fit(smoothing_level=best_alpha, optimized=False)

    y_pred_val = final_model.forecast(step)  # Predicciones en validación
    y_pred_test = final_model.forecast(len(test))  # Predicciones en test
    
    mae_val = mean_absolute_error(val, y_pred_val)

    figura_seshw = plot_model(train, val, test, y_pred_val, y_pred_test, title)

    return y_pred_val, y_pred_test, figura_seshw


pred_HW1_val_pm10, pred_HW1_test_pm10, figura_HW1 = ses_model_tuning(train_ts[-200:], val_ts, test_ts, step=len(val_ts))

print(type(test_ts))  # Verifica el tipo de test_ts
print(type(pred_HW1_test_pm10))  # Verifica el tipo de pred_HW1_test_pm10

# Calcular residuos
residuals_HW1 = test_ts - pred_HW1_test_pm10
# Prueba de Ljung-Box
ljung_box_results_HW1 = acorr_ljungbox(residuals_HW1)
ljung_box_pval_HW1 = ljung_box_results_HW1['lb_pvalue'].values[0]
# Evaluar resultados del test
if ljung_box_pval_HW1 > 0.05:
    print('No se rechaza H0: los residuales son independientes (no correlacionados).')
else:
    print('Se rechaza H0: hay autocorrelación en los residuales.')

# Test de Jarque-Bera
jarque_bera_results_HW1 = jarque_bera(residuals_HW1)
jarque_bera_pval_HW1 = jarque_bera_results_HW1[1]  # Accede al p-valor
#print('Jarque-Bera p-value: %f' % jarque_bera_pval)
# Evaluar resultados del test
if jarque_bera_pval_HW1 > 0.05:
    print('No se rechaza H0: los residuales siguen una distribución normal.')
else:
    print('Se rechaza H0: los residuales no siguen una distribución normal.')


#Calcular métricas 

T0 = len(test_ts)
yh0 = test_ts.copy().values
prederr0 = yh0 - pred_HW1_test_pm10
SSE0 = sum(prederr0**2)
MAPE0 = round(100 * sum(abs(prederr0 / yh0)[yh0 != 0]) / T0,2)
MAD0 = round(sum(abs(prederr0)) / T0,2)
MSD0 = round(sum(prederr0**2) / T0,2)
r2 = round((r2_score(yh0, pred_HW1_test_pm10)*100),2)
ret0 = pd.DataFrame({
        "SSE": [SSE0],
        "MAPE": [f"{MAPE0}%"],
        "MAD": [MAD0],
        "MSD": [MSD0],
        "R2": [f"{r2}%"]
})
ret0.reset_index(drop=True, inplace=True)




#*********************************************************************** FUNCION MLP   **************************************************************************************************

 
model_path = 'PRSA_data_PM10_MLP_weights.07-0.0051.keras'
print(f"Cargando modelo desde: {model_path}")
MLP_model = load_model(model_path)


# Hacer predicciones para val
pred_MLP_val_pm10 = MLP_model.predict(X_val)
pred_MLP_val_pm10 = np.ravel(pred_MLP_val_pm10)
pred_MLP_val_pm10 = pd.Series(pred_MLP_val_pm10, index=data_val_plot.index)


# Hacer predicciones para test
pred_MLP_test_pm10 = MLP_model.predict(X_test)
pred_MLP_test_pm10 = np.ravel(pred_MLP_test_pm10)
pred_MLP_test_pm10 = pd.Series(pred_MLP_test_pm10, index=data_test_plot.index)


# Calcular residuos
residuals_MPL = data_test_plot - pred_MLP_test_pm10
# Prueba de Ljung-Box
ljung_box_results_MLP = acorr_ljungbox(residuals_MPL)
ljung_box_pval_MLP = ljung_box_results_MLP['lb_pvalue'].values[0]
# Evaluar resultados del test
if ljung_box_pval_MLP > 0.05:
    print('No se rechaza H0: los residuales son independientes (no correlacionados).')
else:
    print('Se rechaza H0: hay autocorrelación en los residuales.')

# Test de Jarque-Bera
jarque_bera_results_MLP = jarque_bera(residuals_MPL)
jarque_bera_pval_MLP = jarque_bera_results_MLP[1]  # Accede al p-valor
#print('Jarque-Bera p-value: %f' % jarque_bera_pval)
# Evaluar resultados del test
if jarque_bera_pval_MLP > 0.05:
    print('No se rechaza H0: los residuales siguen una distribución normal.')
else:
    print('Se rechaza H0: los residuales no siguen una distribución normal.')


#Calcular métricas 

T = len(data_test_plot)
yh = data_test_plot.copy().values
prederr = yh - pred_MLP_test_pm10
SSE = sum(prederr**2)
MAPE = round(100 * sum(abs(prederr / yh)[yh != 0]) / T,2)
MAD = round(sum(abs(prederr)) / T,2)
MSD = round(sum(prederr**2) / T,2)
r2 = round((r2_score(yh, pred_MLP_test_pm10)*100),2)
ret1 = pd.DataFrame({
        "SSE": [SSE],
        "MAPE": [f"{MAPE}%"],
        "MAD": [MAD],
        "MSD": [MSD],
        "R2": [f"{r2}%"]
})
ret1.reset_index(drop=True, inplace=True)


#*********************************************************************** FUNCION ANN  **************************************************************************************************

 
ANN_model = load_model('PRSA_data_PM10_ANN_weights.17-32.1529.keras')

# Hacer predicciones para val
pred_ANN_val_pm10 = ANN_model.predict(X_val)
pred_ANN_val_pm10 = np.squeeze(pred_ANN_val_pm10)

# Hacer predicciones para test
pred_ANN_test_pm10 = ANN_model.predict(X_test)
pred_ANN_test_pm10 = np.squeeze(pred_ANN_test_pm10)


# Aplanar el array a 1D
pred_ANN_val_pm10 = np.ravel(pred_ANN_val_pm10)
pred_ANN_val_pm10 = pd.Series(pred_ANN_val_pm10, index=data_val_plot.index)

pred_ANN_test_pm10 = np.ravel(pred_ANN_test_pm10)
pred_ANN_test_pm10 = pd.Series(pred_ANN_test_pm10, index=data_test_plot.index)

# Calcular residuos  ANN
residuals_ANN = data_test_plot - pred_ANN_test_pm10
# Prueba de Ljung-Box
ljung_box_results_ANN = acorr_ljungbox(residuals_ANN)
ljung_box_pval_ANN = ljung_box_results_ANN['lb_pvalue'].values[0]
# Evaluar resultados del test
if ljung_box_pval_ANN > 0.05:
    print('No se rechaza H0: los residuales son independientes (no correlacionados).')
else:
    print('Se rechaza H0: hay autocorrelación en los residuales.')

# Test de Jarque-Bera
jarque_bera_results_ANN = jarque_bera(residuals_ANN)
jarque_bera_pval_ANN = jarque_bera_results_ANN[1]  # Accede al p-valor
#print('Jarque-Bera p-value: %f' % jarque_bera_pval)
# Evaluar resultados del test
if jarque_bera_pval_ANN > 0.05:
    print('No se rechaza H0: los residuales siguen una distribución normal.')
else:
    print('Se rechaza H0: hay autocorrelación en los residuales.')


#Calcular métricas 
T = len(data_test_plot)
yh_ann = data_test_plot.copy().values
prederr_ann = yh_ann - pred_ANN_test_pm10
SSE_ANN = sum(prederr_ann**2)
MAPE_ANN = round(100 * sum(abs(prederr_ann / yh_ann)[yh_ann != 0]) / T,2)
MAD_ANN = round(sum(abs(prederr_ann)) / T,2)
MSD_ANN = round(sum(prederr_ann**2) / T,2)
r2_ANN = round((r2_score(yh_ann, pred_ANN_test_pm10)*100),2)
ret2 = pd.DataFrame({
        "SSE": [SSE_ANN],
        "MAPE": [f"{MAPE_ANN}%"],
        "MAD": [MAD_ANN],
        "MSD": [MSD_ANN],
        "R2": [f"{r2_ANN}%"]
})
ret2.reset_index(drop=True, inplace=True)

# *********************************************************************** FUNCION LSTM  **************************************************************************************************

 
LSTM_model = load_model('PRSA_data_PM10_LSTM_weights.11-0.0051.keras')

# Hacer predicciones para val
pred_LSTM_val_pm10 = LSTM_model.predict(X_val)
pred_LSTM_val_pm10 = np.squeeze(pred_LSTM_val_pm10)
pred_LSTM_val_pm10 = pd.Series(pred_LSTM_val_pm10, index=data_val_plot.index)

# Hacer predicciones para test
pred_LSTM_test_pm10 = LSTM_model.predict(X_test)
pred_LSTM_test_pm10 = np.squeeze(pred_LSTM_test_pm10)
pred_LSTM_test_pm10 = pd.Series(pred_LSTM_test_pm10, index=data_test_plot.index)


# Calcular residuos  LSTM
residuals_LSTM = data_test_plot - pred_LSTM_test_pm10
# Prueba de Ljung-Box
ljung_box_results_LSTM = acorr_ljungbox(residuals_LSTM)
ljung_box_pval_LSTM = ljung_box_results_LSTM['lb_pvalue'].values[0]
# Evaluar resultados del test
if ljung_box_pval_LSTM > 0.05:
    print('No se rechaza H0: los residuales son independientes (no correlacionados).')
else:
    print('Se rechaza H0: hay autocorrelación en los residuales.')

# Test de Jarque-Bera
jarque_bera_results_LSTM = jarque_bera(residuals_LSTM)
jarque_bera_pval_LSTM = jarque_bera_results_LSTM[1]  # Accede al p-valor
#print('Jarque-Bera p-value: %f' % jarque_bera_pval)
# Evaluar resultados del test
if jarque_bera_pval_LSTM > 0.05:
    print('No se rechaza H0: los residuales siguen una distribución normal.')
else:
    print('Se rechaza H0: hay autocorrelación en los residuales.')


#Calcular métricas 
T = len(data_test_plot)
yh_lstm = data_test_plot.copy().values
prederr_lstm = yh_lstm - pred_LSTM_test_pm10
SSE_LSTM = sum(prederr_lstm**2)
MAPE_LSTM = round(100 * sum(abs(prederr_lstm / yh_lstm)[yh_lstm != 0]) / T,2)
MAD_LSTM = round(sum(abs(prederr_lstm)) / T,2)
MSD_LSTM = round(sum(prederr_lstm**2) / T,2)
r2_LSTM = round((r2_score(yh_lstm, pred_LSTM_test_pm10)*100),2)
ret3 = pd.DataFrame({
        "SSE": [SSE_LSTM],
        "MAPE": [f"{MAPE_LSTM}%"],
        "MAD": [MAD_LSTM],
        "MSD": [MSD_LSTM],
        "R2": [f"{r2_LSTM}%"]
})
ret3.reset_index(drop=True, inplace=True)


#***********************************************************************  FUNCION MODELO DE EMSABLE APLICADO DSE-XGB  **************************************************************************************************

 
# Apilamiento de las predicciones
X_trainGBoots = np.column_stack((pred_LSTM_val_pm10, pred_ANN_val_pm10))

# Y data 
y_train_XGBoots = data_val_plot.copy()
y_val_XGBoots = data_test_plot.iloc[:12] 
y_test_XGBoots = data_test_plot.iloc[12:] 

# Entrenar el modelo XGBoost
xgb_meta_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.001)
xgb_meta_model.fit(X_trainGBoots, y_train_XGBoots)  # Suponiendo que y_train es tu variable objetivo.


# Hacer predicciones con el modelo meta
final_val_preds = xgb_meta_model.predict(X_trainGBoots)
pred_XGBOOTS_val_pm10 = final_val_preds[:12] 

pred_XGBOOTS_test_pm10 = final_val_preds[12:]
data_test_plot_XGBOOTS = data_test_plot[12:]  


# Calcular residuos  LSTM
residuals_XGBOOTS = data_test_plot_XGBOOTS - pred_XGBOOTS_test_pm10
# Prueba de Ljung-Box
ljung_box_results_XGBOOTS = acorr_ljungbox(residuals_XGBOOTS)
ljung_box_pval_XGBOOTS  = ljung_box_results_XGBOOTS ['lb_pvalue'].values[0]
# Evaluar resultados del test
if ljung_box_pval_XGBOOTS > 0.05:
    print('No se rechaza H0: los residuales son independientes (no correlacionados).')
else:
    print('Se rechaza H0: hay autocorrelación en los residuales.')

# Test de Jarque-Bera
jarque_bera_results_XGBOOTS = jarque_bera(residuals_XGBOOTS)
jarque_bera_pval_XGBOOTS = jarque_bera_results_XGBOOTS [1]  # Accede al p-valor
#print('Jarque-Bera p-value: %f' % jarque_bera_pval)
# Evaluar resultados del test
if jarque_bera_pval_XGBOOTS  > 0.05:
    print('No se rechaza H0: los residuales siguen una distribución normal.')
else:
    print('Se rechaza H0: hay autocorrelación en los residuales.')


#Calcular métricas 
T = len(data_test_plot_XGBOOTS)
yh_xgboots = data_test_plot_XGBOOTS.copy().values
prederr_xgboots = yh_xgboots - pred_XGBOOTS_test_pm10
SSE_XGBOOTS = sum(prederr_xgboots**2)
MAPE_XGBOOTS = round(100 * sum(abs(prederr_xgboots / yh_xgboots)[yh_xgboots != 0]) / T,2)
MAD_XGBOOTS = round(sum(abs(prederr_xgboots)) / T,2)
MSD_XGBOOTS = round(sum(prederr_xgboots**2) / T,2)
r2_XGBOOTS = round((r2_score(yh_xgboots, pred_XGBOOTS_test_pm10)*100),2)
ret4 = pd.DataFrame({
        "SSE": [SSE_XGBOOTS],
        "MAPE": [f"{MAPE_XGBOOTS}%"],
        "MAD": [MAD_XGBOOTS],
        "MSD": [MSD_XGBOOTS],
        "R2": [f"{r2_XGBOOTS}%"]
})
ret4.reset_index(drop=True, inplace=True)


#***********************************************************************  FUNCION MODELO SUAVIZACIÓN EXPONENCIAL DOBLE*************************************************************************************




##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                                                CREAR APLICACIÓN DASH                                                                                           #
#                                                                                                                                                                                #
##################################################################################################################################################################################

# Definir la URL de MathJax
mathjax_url = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'

# Configurar la aplicación Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.LITERA],
    external_scripts=[mathjax_url],  # Usar external_scripts para incluir MathJax
    suppress_callback_exceptions=True
)

server = app.server

# Definir la escala de colores
colorscale = cl.scales['9']['qual']['Paired']

# Registrar la página
dash.register_page(
    __name__,
    path='/book_now',
    title="How neural network learns",
    description="How neural network learns",
    image='hogar.png'
)



##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                                            CONFIGURACIÓN DE GRÁFICOS                                                                                           #
#                                                                                                                                                                                #
##################################################################################################################################################################################


#Función Gráfico en Blanco para luego rellenar con la gráfica deseada (temas de compatibilidad DASH)
def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    
    return fig



##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                                            CÓDIFICACIÓN DE IMAGENES LOCALES                                                                                    #
#                                                                                                                                                                                #
##################################################################################################################################################################################

# Codificar la imagen BOOK NOW PAGE
Home = 'assets/Home2.png'
with open(Home, 'rb') as image_file:
    imagen_codificada = base64.b64encode(image_file.read()).decode('utf-8')


#Pintar gráficos 

figura_mlp=plot_model(data_train_plot[-200:], data_val_plot, data_test_plot, pred_MLP_val_pm10, pred_MLP_test_pm10, "Predicciones usando Perceptrón Multicapa")

figura_ann=plot_model(data_train_plot[-200:], data_val_plot, data_test_plot, pred_ANN_val_pm10, pred_ANN_test_pm10, "Predicciones usando Red Neuronal Artificial")

figura_lstm=plot_model(data_train_plot[-200:], data_val_plot, data_test_plot, pred_LSTM_val_pm10, pred_LSTM_test_pm10, "Predicciones usando Memoria a Corto y Largo Plazo")

figure_xgb=plot_model(y_train_XGBoots, y_val_XGBoots, y_test_XGBoots, pred_XGBOOTS_val_pm10, pred_XGBOOTS_test_pm10, "Predicciones usando un modelo de ensamble apilado (DSE-XGB)")



##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                                                  CONTENIDO DE LAYOUT                                                                                           #
#                                                                                                                                                                                #
##################################################################################################################################################################################


layout_book= html.Div([
        dbc.Card(
            dbc.CardBody(
                [dbc.Row([
                    dbc.Col(
                        html.H1("Modelos Predictivos de Series de Tiempo",
                        style={'width': '100%', 'height': 'auto', 'color': "#4f5382",'font-size':'30px'}),
                        className="d-flex align-items-center")]),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody([
                                            html.Br(),
                                            html.Br(),
                                            dbc.Col(
                                            [
                                            dbc.Row([                                                
                                                html.Div([
                                                    html.H5("A continuación se visualizan las predicciones para los datos de validación y prueba del Material Particulado de 10μm", style={"color": "#4f5382",'font-size':'18px'}),
                                                ]),
                                            html.Br(),
                                            ]),
                                            dbc.Row([
                                                    html.Br(), 
                                                    dcc.Loading(
                                                        html.Div([
                                                            dcc.Graph(id='MLP_plot',figure=figura_mlp, style={   #Gráfico de predicciones MLP.
                                                                'width': '80%',
                                                                'height': '50vh',
                                                                'display': 'flex',
                                                                'align-items': 'center',
                                                                'justify-content': 'center',
                                                            })
                                                        ])
                                                    ),
                                                    html.Br(),  # Corregido aquí                                              
                                                    ],
                                                    align="center",
                                                ),
                                            html.Br(),
                                            dbc.Row([
                                                    html.Br(), 
                                                    dcc.Loading(
                                                        html.Div([
                                                        dcc.Graph(id="grafica_ANN",figure=figura_ann, style={   #Gráfico de predicciones MLP.
                                                                'width': '80%',
                                                                'height': '50vh',
                                                                'display': 'flex',
                                                                'align-items': 'center',
                                                                'justify-content': 'center',
                                                            }),# id de gráfico de barra por años  
                                                        ])
                                                    ),
                                                    html.Br(),  # Corregido aquí                                              
                                                    ],
                                                    align="center",
                                                ),
                                            html.Br(),
                                            dbc.Row([
                                                    html.Br(), 
                                                    dcc.Loading(
                                                        html.Div([
                                                        dcc.Graph(id="grafica_LSTM",figure=figura_lstm, style={   #Gráfico de predicciones MLP.
                                                                'width': '80%',
                                                                'height': '50vh',
                                                                'display': 'flex',
                                                                'align-items': 'center',
                                                                'justify-content': 'center',
                                                            }),# id de gráfico de barra por años  
                                                        ])
                                                    ),
                                                    html.Br(),  # Corregido aquí                                              
                                                    ],
                                                    align="center",
                                                ),
                                            html.Br(),
                                            dbc.Row([
                                                    html.Br(), 
                                                    dcc.Loading(
                                                        html.Div([
                                                        dcc.Graph(id="grafica_DSE_XGB",figure=figure_xgb, style={   #Gráfico de predicciones MLP.
                                                                'width': '80%',
                                                                'height': '50vh',
                                                                'display': 'flex',
                                                                'align-items': 'center',
                                                                'justify-content': 'center',
                                                            }),# id de gráfico de barra por años  
                                                        ])
                                                    ),
                                                    html.Br(),  # Corregido aquí                                              
                                                    ],
                                                    align="center",
                                                ), 
                                            html.Br(), 
                                            html.Br(),
                                            html.Br(),
                                            html.Br(), 
                                            html.Br(),                         
                                            ],
                                            align="center",
                                        ),                                                                                                                        
                                        ]
                                    ),
                                ),
                                sm=6,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [ dbc.Row([
                                                    html.Br(), 
                                                    dcc.Loading(
                                                        html.Div([
                                                            dcc.Graph(id='HW1_plot',figure=figura_HW1, style={   #Gráfico de predicciones MLP.
                                                                'width': '80%',
                                                                'height': '50vh',
                                                                'display': 'flex',
                                                                'align-items': 'center',
                                                                'justify-content': 'center',
                                                            })
                                                        ])
                                                    ),
                                                    html.Br(),  # Corregido aquí                                              
                                                    ],
                                                    align="center",
                                                ),
                                            html.Br(),
                                            html.Hr(),    
                                            html.H1("Métricas de Evaluación de Modelos",style={"color" : "#4f5382",'font-size':'18px'}),
                                            html.Hr(),
                                            html.Br(),
                                            dbc.Col([
                                                    html.H1("Suavización Exponencial HoltWinters 2Do Orden (SES HW)",style={"color" : "#4f5382",'font-size':'16px'}),
                                                    html.Hr(),
                                                    html.Br(),                                                    
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/bubbles/100/bar-chart.png", alt="bar-chart", style={'width':'60px'}),
                                                            html.Div(children=str(round(ljung_box_pval_HW1,3)),style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # ljungbox
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Dado que el p-valor es < 0,05 se rechaza H0, por tanto, se puede afirmar que existe autocorrelación en los residuales",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/fluency/48/high-risk--v1.png", alt="high-risk--v1", style={'width':'50px'}),
                                                            html.Div(children=str(round(jarque_bera_pval_HW1,3)),style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # Jaque Bera
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Dado que el p-valor es < 0,05 Se rechaza H0, por tanto, se puede afirmar que los residuales no siguen una distribución normal.",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),
                                                
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/external-filled-outline-geotatah/64/external-brand-brand-positioning-filled-outline-filled-outline-geotatah-10.png", 
                                                                     alt="external-brand-brand-positioning-filled-outline-filled-outline-geotatah-10", style={'width':'50px'}),
                                                            html.Div(children=f"{round(MAD0, 3)} μm",style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # MAD
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Este valor corresponde a la media de las desviaciones absolutas de las predicciones con respecto a su media",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),]),
                                                    html.Br(), #Hasta aqui copiar
                                                    html.Br(),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.H1("Perceptrones Multicapa (MLP)",style={"color" : "#4f5382",'font-size':'16px'}),
                                                    html.Hr(),
                                                     html.Br(),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/bubbles/100/bar-chart.png", alt="bar-chart", style={'width':'60px'}),
                                                            html.Div(children=str(round(ljung_box_pval_MLP,3)),style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # ljungbox
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Dado que el p-valor es < 0,05 se rechaza H0, por tanto, se puede afirmar que existe autocorrelación en los residuales",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/fluency/48/high-risk--v1.png", alt="high-risk--v1", style={'width':'50px'}),
                                                            html.Div(children=str(round(jarque_bera_pval_MLP,3)),style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # Jaque Bera
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Dado que el p-valor es > 0,05 No se rechaza H0, por tanto, se puede afirmar que los residuales siguen una distribución normal.",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/external-filled-outline-geotatah/64/external-brand-brand-positioning-filled-outline-filled-outline-geotatah-10.png", 
                                                                     alt="external-brand-brand-positioning-filled-outline-filled-outline-geotatah-10", style={'width':'50px'}),
                                                            html.Div(children=f"{round(MAD, 3)} μm",style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # MAD
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Este valor corresponde a la media de las desviaciones absolutas de las predicciones con respecto a su media",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),
                                                    html.Br(),
                                                    html.Br(), 
                                                    dbc.Col([
                                                    html.H1("Red Neuronal Artificial (ANN)",style={"color" : "#4f5382",'font-size':'16px'}),
                                                    html.Hr(),
                                                    html.Br(),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/bubbles/100/bar-chart.png", alt="bar-chart", style={'width':'60px'}),
                                                            html.Div(children=str(round(ljung_box_pval_ANN,3)),style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # ljungbox
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Dado que el p-valor es < 0,05 se rechaza H0, por tanto, se puede afirmar que existe autocorrelación en los residuales",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/fluency/48/high-risk--v1.png", alt="high-risk--v1", style={'width':'50px'}),
                                                            html.Div(children=str(round(jarque_bera_pval_ANN,3)),style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # Jaque Bera
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Dado que el p-valor es > 0,05 No se rechaza H0, por tanto, se puede afirmar que los residuales siguen una distribución normal.",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/external-filled-outline-geotatah/64/external-brand-brand-positioning-filled-outline-filled-outline-geotatah-10.png", 
                                                                     alt="external-brand-brand-positioning-filled-outline-filled-outline-geotatah-10", style={'width':'50px'}),
                                                            html.Div(children=f"{round(MAD_ANN, 3)} μm",style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # MAD
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Este valor corresponde a la media de las desviaciones absolutas de las predicciones con respecto a su media",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),]),
                                                    html.Br(), #Hasta aqui copiar
                                                    html.Br(),
                                                    dbc.Col([
                                                    html.H1("Memoria a Corto y Largo Plazo (LSTM)",style={"color" : "#4f5382",'font-size':'16px'}),
                                                    html.Hr(),
                                                    html.Br(),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/bubbles/100/bar-chart.png", alt="bar-chart", style={'width':'60px'}),
                                                            html.Div(children=str(round(ljung_box_pval_LSTM,3)),style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # ljungbox
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Dado que el p-valor es < 0,05 se rechaza H0, por tanto, se puede afirmar que existe autocorrelación en los residuales",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/fluency/48/high-risk--v1.png", alt="high-risk--v1", style={'width':'50px'}),
                                                            html.Div(children=str(round(jarque_bera_pval_LSTM,3)),style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # Jaque Bera
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Dado que el p-valor es > 0,05 No se rechaza H0, por tanto, se puede afirmar que los residuales siguen una distribución normal.",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/external-filled-outline-geotatah/64/external-brand-brand-positioning-filled-outline-filled-outline-geotatah-10.png", 
                                                                     alt="external-brand-brand-positioning-filled-outline-filled-outline-geotatah-10", style={'width':'50px'}),
                                                            html.Div(children=f"{round(MAD_LSTM, 3)} μm",style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # MAD
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Este valor corresponde a la media de las desviaciones absolutas de las predicciones con respecto a su media",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),]),
                                                    html.Br(), #Hasta aqui copiar
                                                    html.Br(),
                                                    dbc.Col([
                                                    html.H1("Modelo de Emsamble Aplilado (DSE-XGB)",style={"color" : "#4f5382",'font-size':'16px'}),
                                                    html.Hr(),
                                                    html.Br(),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/bubbles/100/bar-chart.png", alt="bar-chart", style={'width':'60px'}),
                                                            html.Div(children=str(round(ljung_box_pval_XGBOOTS,3)),style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # ljungbox
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Dado que el p-valor es > 0,05 No se rechaza H0, por tanto, se puede afirmar que no existe autocorrelación en los residuales",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/fluency/48/high-risk--v1.png", alt="high-risk--v1", style={'width':'50px'}),
                                                            html.Div(children=str(round(jarque_bera_pval_XGBOOTS,3)),style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # Jaque Bera
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Dado que el p-valor es > 0,05 No se rechaza H0, por tanto, se puede afirmar que los residuales siguen una distribución normal.",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),
                                                    dbc.Row([
                                                         dbc.Col([  
                                                        dcc.Loading(                                        
                                                        html.Div([
                                                            html.Img(src="https://img.icons8.com/external-filled-outline-geotatah/64/external-brand-brand-positioning-filled-outline-filled-outline-geotatah-10.png", 
                                                                     alt="external-brand-brand-positioning-filled-outline-filled-outline-geotatah-10", style={'width':'50px'}),
                                                            html.Div(children=f"{round(MAD_XGBOOTS, 3)} μm",style={"color" : "#4f5382",'font-size':'16px'}),                                                                                                                                                                                                                                  
                                                        ], className="d-flex flex-column align-items-center")),  # MAD
                                                         ]), 
                                                         dbc.Col(
                                                         html.H1("Este valor corresponde a la media de las desviaciones absolutas de las predicciones con respecto a su media",style={"color" : "#4f5382",'font-size':'16px'})
                                                         ),
                                                    ]),]),
                                                    html.Br(), #Hasta aqui copiar
                                                    ],
                                                    align="center",                                                    
                                                ),                                                                                       
                                                ]),
                                        ]
                                    ),
                                ),
                                sm=6,
                            ),
                            html.Br(), 
                            html.Br(), 
                        ],
                        align="center",
                    ),
                ]
            ),
            style={
                'overflowY': 'auto',      # Permite desplazamiento si es necesario
                'height': '90vh',         # Ajusta la altura del CardBody para ocupar más de la pantalla
                'padding-bottom': '0px'         # Ajusta el padding para mejor presentación
                }
        ),                         
            ],style={
        'padding-bottom': '0',    # Reduce padding si no es necesario
        'height': '90vh', # Asegura que todo el contenido ocupe la altura completa
    }
)


##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                                                              CALLBACKS                                                                                         #
#                                                                                                                                                                                #
##################################################################################################################################################################################



#Callback para gráfico ##############################################################







#Call Back para lasso ################################################################################ 


#call Back para Ridge ################################################################################ 



#CallBack para SVM  ################################################################################ 



#Callback para estimaciones puntuales  ################################################################################ 




if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(debug=True, host='0.0.0.0', port=9000) # <- To Dockerize the Dash