##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                                              IMPORTACION DE LIBRERÍAS                                                                                          #
#                                                                                                                                                                                #
##################################################################################################################################################################################

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


# Load sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from math import *
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,RobustScaler
from sklearn.compose import ColumnTransformer


#Load Dash
import dash 
import math
from dash import Dash, Input, Output, callback, dash_table, html, dcc 
import pandas as pd
import dash_bootstrap_components as dbc
import colorlover as cl
import psycopg2
import pandas as pd 
import kaleido
from sqlalchemy import create_engine
import numpy as np
import dash
import psycopg2
import pandas as pd 
import kaleido
from sqlalchemy import create_engine
import numpy as np
import plotly.graph_objects as go
import sklearn.metrics
import plotly.express as px
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import base64
import urllib
import json
import csv


import warnings
warnings.filterwarnings("ignore")


##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                                              TRATAMIENTO DE DATOS                                                                                              #
#                                                                                                                                                                                #
##################################################################################################################################################################################

#df = pd.read_csv('C:/Users/kathy/Desktop/ml/app/housingcaliforia.csv')

df = pd.read_csv('housingcaliforia.csv') #RUTA DE DIRECTORIO DE TRABAJO DE CONTENEDOR

# Renombrar variables 
df_r = df.rename(columns={
    'longitude': 'long', 
    'latitude': 'lati', 
})

#Imputación de NaN 
df_r['total_bedrooms'] = df_r['total_bedrooms'].fillna(df_r['total_bedrooms'].mean())

# Datos Mapa 

#lectura de datos 
df = pd.read_csv("data_2017_2022.csv", sep=";")
df_1 = df.copy()
df_1["medicion"]=df_1["medicion"].str.replace(",", ".").astype("float") # convertir variable medición a númerica
df_1['fecha'] = pd.to_datetime(df_1['fecha']) # convertir variable fecha a datatime
df_1 = df_1.pivot(index=["fecha", "estacion"],columns= "variable", values="medicion").reset_index()

# Renombrar variables 
df_1 = df_1.rename(columns={    
    'variable': 'var',
    'black_carbon': 'BC',
    'direccion_viento': 'WD',
    'humedad': 'HR',
    'lluvia': 'RF',
    'presion': 'P',
    'radiacion_solar': 'RS',
    'temperatura': 'AT',
    'temperatura_10_m': 'AT_10_m',
    'uv-pm': 'UV',
    'velocidad_viento': 'WS',
    'h2s': 'H2S',
    'no2': 'NO2',
    'o3': 'O3',
    'so2': 'SO2',
    'pm10': 'PM10',
    'pm25': 'PM2.5'  
})

#Filtro de Datos 

df_t = df_1.copy()
df_t = df_t.drop(columns=["NO2", "BC", "HR", "RF", "P", "RS", "AT", "AT_10_m", "UV", "WS", "H2S", "O3", "SO2", "PM2.5", "WD"])
df_t = df_t.dropna(subset=['PM10'])

df_t.info()
print(df_t["estacion"].unique())


#Mapa de Cali (Coordenadas) 

pd.set_option('display.precision', 4)
# Definir el diccionario de coordenadas
coords = {
    'base_aerea': {'Latitud': 3.4343, 'Longitud': -76.5226},
    'canaveralejo': {'Latitud': 3.4267, 'Longitud': -76.5239},
    'compartir': {'Latitud': 3.4295, 'Longitud': -76.5384},
    'era_obrero': {'Latitud': 3.4045, 'Longitud': -76.5525},
    'ermita': {'Latitud': 3.4352, 'Longitud': -76.5416},
    'flora': {'Latitud': 3.4350, 'Longitud': -76.5160},
    'pance': {'Latitud': 3.4060, 'Longitud': -76.5780},
    'transitoria': {'Latitud': 3.4355, 'Longitud': -76.5371},
    'univalle': {'Latitud': 3.4189, 'Longitud': -76.5641}
}

# Agregar las columnas de Latitud y Longitud al DataFrame df_t
df_t['Latitud'] = df_t['estacion'].map(lambda x: coords[x]['Latitud'])
df_t['Longitud'] = df_t['estacion'].map(lambda x: coords[x]['Longitud'])









##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                               DEFINICIÓN DE FUNCIONES MODELOS ML                                                                                               #
#                                                                                                                                                                                #
##################################################################################################################################################################################




#*********************************************************************** FUNCION knn  **************************************************************************************************

def modeloknn(data): 

#data transformation    
    
    data_dummies = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)# Crear variables dummy solo para las columnas de tipo objeto
    datosmodknn = data_dummies.astype(int)# Convertir las columnas de dummies a tipo entero
    # Extraer columnas en formato array
    X = datosmodknn.drop('median_house_value', axis=1).to_numpy()  # Características
    Y = datosmodknn['median_house_value'].to_numpy()  # Target

    #Confi tamaño y semilla datos train y test
    validation_size = 0.30
    seed = 7

    #división de la data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                test_size=validation_size, random_state=seed, shuffle=True)

    scaler = StandardScaler()
    steps_knn = [("scale", scaler)]

    pipeline_knn = Pipeline(steps_knn)
    x_train_knn = pipeline_knn.fit_transform(X_train)
    x_test_knn = pipeline_knn.transform(X_test)  # garantizamos que los datos de prueba se escalen usando la misma media y desviación estándar que se calcularon a partir del conjunto de entrenamiento.


    # Definir parámetros para GridSearchCV
    parameters = {'n_neighbors': range(1, 20), 'weights': ['uniform', 'distance']}

    # Crear el modelo y ajustar GridSearchCV
    bestmodel_knn = GridSearchCV(KNeighborsRegressor(), parameters, cv=3)
    bestmodel_knn.fit(x_train_knn, Y_train)

    #Extraer mejores parametros y almacenar en un directorio llamado parameters
    parameters = bestmodel_knn.best_params_

    #Extraer modelo 
    model = KNeighborsRegressor(**parameters) 

    # Predecir utilizando el modelo ajustado y calcular métricas
    y_test_preds = bestmodel_knn.predict(x_test_knn)

    # Calcular el R^2
    r2_test = r2_score(Y_test, y_test_preds)

    y_test_preds_medio = y_test_preds[~np.isnan(y_test_preds)] 

    y_test_preds_medio = y_test_preds_medio.mean()

    
    return  y_test_preds_medio, r2_test
 


# *********************************************************************** FUNCION RL  **************************************************************************************************

def modelorl(data): 

#data transformation    
    data = df_r.copy()
    data_dummies = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)# Crear variables dummy solo para las columnas de tipo objeto
    datosmodrl = data_dummies.astype(int)# Convertir las columnas de dummies a tipo entero
    # Extraer columnas en formato array
    X = datosmodrl.drop('median_house_value', axis=1).to_numpy()  # Características
    Y = datosmodrl['median_house_value'].to_numpy()  # Target

    #Confi tamaño y semilla datos train y test
    validation_size = 0.30
    seed = 7

    #división de la data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                test_size=validation_size, random_state=seed, shuffle=True)

    # Estandarización de los datos    
    scaler = StandardScaler()       
    steps_rl = [
        ("scale", scaler)
    ]

    pipeline_rl = Pipeline(steps_rl)
    x_train_rl = pipeline_rl.fit_transform(X_train)
    x_test_rl = pipeline_rl.fit_transform(X_test)

    # Definir parámetros para linear regression 
    modelo_rl = linear_model.LinearRegression()
    modelo_rl.fit(x_train_rl, Y_train)

    y_test_preds = modelo_rl.predict(x_test_rl)

    # Calcular el R^2
    r2_test = r2_score(Y_test, y_test_preds)

    y_test_preds_medio = y_test_preds[~np.isnan(y_test_preds)] 

    y_test_preds_medio = y_test_preds_medio.mean()

    
    return  y_test_preds_medio, r2_test




#*********************************************************************** FUNCION SVM BRF   **************************************************************************************************

def modeloSVM(data): 

#data transformation    
    data = df_r.copy()
    data_dummies = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)# Crear variables dummy solo para las columnas de tipo objeto
    datosmosvm = data_dummies.astype(int)# Convertir las columnas de dummies a tipo entero
    # Extraer columnas en formato array
    X = datosmosvm.drop('median_house_value', axis=1).to_numpy()  # Características
    Y = datosmosvm['median_house_value'].to_numpy()  # Target

    #Confi tamaño y semilla datos train y test
    validation_size = 0.30
    seed = 7

    #división de la data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                test_size=validation_size, random_state=seed, shuffle=True)

    # Estandarización de los datos    
    scaler = StandardScaler()       
    steps_svm = [
        ("scale", scaler)
    ]

    pipeline_svm = Pipeline(steps_svm)
    x_train_svm = pipeline_svm.fit_transform(X_train)
    x_test_svm = pipeline_svm.fit_transform(X_test)

    # Definir parámetros para Suport Vector Machine BRF
    modelo_svmrbf = SVR(kernel='rbf', gamma='scale', C=1000000)
    modelo_svmrbf.fit(x_train_svm, Y_train)

    y_train_preds = modelo_svmrbf.predict(x_train_svm)
    mse_train  = mean_squared_error(Y_train, y_train_preds)
    rmse_train = sqrt(mse_train)
    rmse_train

    y_test_preds = modelo_svmrbf.predict(x_test_svm)

    # Calcular el R^2
    r2_test = r2_score(Y_test, y_test_preds)

    y_test_preds_medio = y_test_preds[~np.isnan(y_test_preds)] 

    y_test_preds_medio = y_test_preds_medio.mean()

    
    return  y_test_preds_medio, r2_test

 


#***********************************************************************  FUNCION LASSO   **************************************************************************************************

def modelolasso(data): 

#data transformation    
    data = df_r.copy()
    data_dummies = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)# Crear variables dummy solo para las columnas de tipo objeto
    datosmodlasso = data_dummies.astype(int)# Convertir las columnas de dummies a tipo entero
    # Extraer columnas en formato array
    X = datosmodlasso.drop('median_house_value', axis=1).to_numpy()  # Características
    Y = datosmodlasso['median_house_value'].to_numpy()  # Target

    #Confi tamaño y semilla datos train y test
    validation_size = 0.30
    seed = 7

    #división de la data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                test_size=validation_size, random_state=seed, shuffle=True)

    # Estandarización de los datos    
    scaler = StandardScaler()       
    steps_lasso = [
        ("scale", scaler)
    ]

    pipeline_lasso = Pipeline(steps_lasso)
    x_train_lasso = pipeline_lasso.fit_transform(X_train)
    x_test_lasso = pipeline_lasso.fit_transform(X_test)

    # Definir parámetros para Suport Vector Machine BRF
    modelo_lasso = Lasso(alpha=1)
    modelo_lasso.fit(x_train_lasso, Y_train)

    y_train_preds = modelo_lasso.predict(x_train_lasso)
    mse_train  = mean_squared_error(Y_train, y_train_preds)
    rmse_train = sqrt(mse_train)
    rmse_train

    y_test_preds = modelo_lasso.predict(x_test_lasso)

    # Calcular el R^2
    r2_test = r2_score(Y_test, y_test_preds)

    y_test_preds_medio = y_test_preds[~np.isnan(y_test_preds)] 

    y_test_preds_medio = y_test_preds_medio.mean()

    
    return  y_test_preds_medio, r2_test

 


#***********************************************************************  FUNCION RIDGE   **************************************************************************************************

def modeloridge(data): 

#data transformation    
    data = df_r.copy()
    data_dummies = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)# Crear variables dummy solo para las columnas de tipo objeto
    datosmodridge = data_dummies.astype(int)# Convertir las columnas de dummies a tipo entero
    # Extraer columnas en formato array
    X = datosmodridge.drop('median_house_value', axis=1).to_numpy()  # Características
    Y = datosmodridge['median_house_value'].to_numpy()  # Target

    #Confi tamaño y semilla datos train y test
    validation_size = 0.30
    seed = 7

    #división de la data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                test_size=validation_size, random_state=seed, shuffle=True)

    # Estandarización de los datos    
    scaler = StandardScaler()       
    steps_ridge = [
        ("scale", scaler)
    ]

    pipeline_ridge = Pipeline(steps_ridge)
    x_train_ridge = pipeline_ridge.fit_transform(X_train)
    x_test_ridge = pipeline_ridge.fit_transform(X_test)

    # Definir parámetros para Suport Vector Machine BRF
    modelo_ridge = Ridge(alpha=1)
    modelo_ridge.fit(x_train_ridge, Y_train)

    y_train_preds = modelo_ridge.predict(x_train_ridge)
    mse_train  = mean_squared_error(Y_train, y_train_preds)
    rmse_train = sqrt(mse_train)
    rmse_train

    y_test_preds = modelo_ridge.predict(x_test_ridge)
  
    # Calcular el R^2
    r2_test = r2_score(Y_test, y_test_preds)

    y_test_preds_medio = y_test_preds[~np.isnan(y_test_preds)] 

    y_test_preds_medio = y_test_preds_medio.mean()

    
    return  y_test_preds_medio, r2_test


 


#******************************************  FUNCION NUMERO DE ESPACIOS, DORMITORIOS E INGRESOS MEDIOS FAMILIARES  ******************************************************************

def number_spaces(data, valor_slct_far_sea): 
    # Filtrar los datos según la selección del dropdown
    df_filtered = df_r[df_r['ocean_proximity'] == valor_slct_far_sea]

    t_rooms = df_filtered['total_rooms'].sum()
    t_bedrooms = df_filtered['total_bedrooms'].sum()
    t_homes = df_filtered['households'].sum()
    t_median_inc = df_filtered['median_income'].mean()


    rooms_p = round(t_rooms/t_homes)
    bedrooms_p = round(t_bedrooms/t_homes)
    median_inc_p = round(t_median_inc*10000)
   

    return rooms_p, bedrooms_p, median_inc_p

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

#Cinfiguración de token para visualizar maps 
# Asegúrate de tener un token de Mapbox
#with open(r".mapbox_token", "r") as file:
#    mapbox_token = file.read().strip()

#px.set_mapbox_access_token(mapbox_token)

# Calcular la media de precios de las casas para cada coordenada única
df_grouped = df_r.groupby(['lati', 'long'], as_index=False)['median_house_value'].mean()
# Formatear los valores de 'median_house_value' con separadores de miles y añadir "USD"
df_grouped['formatted_price'] = df_grouped['median_house_value'].apply(lambda x: f"{x:,.0f} USD")


# Define los intervalos, colores y clasificaciones
def get_color_and_classification(pm10):
    if pm10 <= 50:
        return 'green', 'Bueno'
    elif pm10 <= 100:
        return 'yellow', 'Aceptable'
    elif pm10 <= 150:
        return 'orange', 'Dañina a la salud grupos sensibles'
    elif pm10 <= 200:
        return 'red', 'Daño a la salud'
    elif pm10 <= 300:
        return 'purple', 'Muy dañino a la salud'
    elif pm10 <= 500:
        return 'brown', 'Peligroso'
    else:
        return 'grey', 'Extremadamente peligroso'

# Calcula estadísticas de PM10
df_stats = df_t.groupby('estacion').agg({
    'PM10': ['mean']
}).reset_index()

# Renombra las columnas para mayor claridad
df_stats.columns = ['estacion', 'PM10_Media']

# Añadir columnas de color y clasificación basadas en la media de PM10
df_stats[['Color', 'Clasificacion']] = df_stats.apply(lambda row: pd.Series(get_color_and_classification(row['PM10_Media'])), axis=1)

# Une los datos estadísticos con el DataFrame de coordenadas
df_t_merged = df_t.merge(df_stats[['estacion', 'Color', 'Clasificacion', 'PM10_Media']], on='estacion')

# Crear el mapa
mapa = go.Figure()

# Agregar marcadores para cada estación
mapa.add_trace(go.Scattermapbox(
    lat=df_t_merged['Latitud'],
    lon=df_t_merged['Longitud'],
            mode='markers',
            marker=dict(
            size=15,
            color=df_t_merged['Color'],
            symbol ="circle",
    ),
    text=df_t_merged['estacion'],
    textposition='top center',
    customdata=df_t_merged[['PM10_Media', 'Clasificacion']],
    hovertemplate=
    '<b>Estación:</b> %{text}<br>' +
    '<b>Clasificación:</b> %{customdata[1]}<br>' +
    '<b>Media PM10:</b> %{customdata[0]:.2f} µg/m³<br>' +
    '<extra></extra>'
))

# Configuración del mapa
mapa.update_layout(
    mapbox=dict(
        style="open-street-map",
        zoom=11,
        center=dict(lat=df_t_merged['Latitud'].mean(), lon=df_t_merged['Longitud'].mean()),
    ),
    title="Ubicación de Estaciones Meteorológicas en Cali",
    title_x=0.5,
    height=600,
    margin={"r":0,"t":40,"l":0,"b":0}
)

# Layout de la aplicación
app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Ubicación Estaciones Metereológicas"),
                dcc.Graph(figure=mapa)
            ])
        ])
    ])
])


# Crear el mapa
#mapa = px.scatter_mapbox(df_grouped, lat='lati', lon='long', size='median_house_value',
#                         color='median_house_value', color_continuous_scale=px.colors.sequential.Purp,
#                         size_max=10, zoom=4, mapbox_style="carto-positron",
 #                        title="Average property price in California",
  #                       labels={'median_house_value': 'Property Price'})

# Añadir la leyenda personalizada al mapa
#mapa.update_traces(marker=dict(size=8),
 #                  hovertemplate='<b>Property Price:</b> %{customdata[0]}<extra></extra>',
  #                 customdata=df_grouped[['formatted_price']])

# Centrar el título del mapa
#mapa.update_layout(title_x=0.5)



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
                        html.H1("Get a quote with us",
                        style={'width': '100%', 'height': 'auto', 'color': "#4f5382",'font-size':'30px'}),
                        className="d-flex align-items-center"), 
                        dbc.Col(
                        html.Div(
                            html.Img(
                                src=f'data:image/png;base64,{imagen_codificada}',
                                style={'width': '100%', 'height': '100%', 'float': 'center'} 
                                    ),),
                                ),]),
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
                                                    html.H5("How far is the property from the sea?", style={"color": "#4f5382",'font-size':'16px'}),
                                                    dcc.Dropdown(
                                                        id="slct_far_sea",
                                                        options=[
                                                            {"label": "Near the bay", "value": "NEAR BAY"},
                                                            {"label": "Far from the bay", "value": "INLAND"},
                                                            {"label": "With a view of the bay", "value": "NEAR OCEAN"},  # Corregido el error de sintaxis
                                                        ],
                                                        multi=False,
                                                        value='NEAR BAY',
                                                        style={'width': "100%", 'font-size': '15px', 'border-radius': '30px'}
                                                    ),
                                                ]),
                                            html.Br(),
                                            html.Br(),
                                            html.Br(),
                                            html.Br(),
                                                dcc.Loading( 
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-tower-spain-inipagistudio-lineal-color-inipagistudio.png", alt="external-analytics-leadership-justicon-lineal-color-justicon", style={'width':'40px'}),
                                                        html.Div(id='rooms_p_output',style={"color" : "#4f5382",'font-size':'16px'}), 
                                                        ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),
                                                    html.Br(),
                                                    dcc.Loading(                                                                                                                                                  
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-bed-house-keeping-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                        html.Div(id='bedrooms_p_output',style={"color" : "#4f5382",'font-size':'16px'}),  
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),
                                                    html.Br(),
                                                    html.Br(),
                                                    dcc.Loading( 
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-money-management-financial-literacy-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                        html.Div(id='median_inc_p_output',style={"color" : "#4f5382",'font-size':'16px'}),  
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),
                                                    html.Br(),
                                                    html.Br(), 
                                            ]),
                                            html.Br(),
                                            html.Br(),
                                            html.Br(),
                                            html.Br(),
                                            html.Br(),
                                            html.Br(),
                                            html.Br(),
                                            html.Br(),
                                            html.Br(),
                                            html.Br(),
                                            html.Br(),

                                            dbc.Row([
                                                dcc.Loading(                                             
                                                    html.Div(
                                                    dcc.Graph(figure=mapa) 
                                                    ,style={
                                                'width': '100%', 
                                                'height': '50vh', 
                                                'display': 'flex', 
                                                'align-items': 'center',  # Centrado vertical
                                                'justify-content': 'center',  # Centrado horizontal
                                                    }
                                                    )),  # Corregido aquí                                              
                                                    ],
                                                    align="center",
                                                ),
                                            html.Br(),
                                            html.Br(),
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
                                sm=8,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H1("Real Estate Price Prediction",style={"color" : "#4f5382",'font-size':'18px'}),
                                            html.Hr(),
                                            html.Br(),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.H1("K-Nearest Neighbors Prediction",style={"color" : "#4f5382",'font-size':'16px'}),
                                                    html.Hr(),  
                                                    dcc.Loading(                                        
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-cash-financial-literacy-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                        html.Div(id='value_predit_knn_test_output',style={"color" : "#4f5382",'font-size':'16px'}),                                                        
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),  # Corregido aquí
                                                    html.Br(),
                                                    dcc.Loading( 
                                                    html.Div([
                                                    html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                    html.Div(id='r2_test_output',style={"color" : "#4f5382",'font-size':'16px'}), 
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),
                                                    html.Br(),    
                                                    html.H1("Linear Regression Prediction",style={"color" : "#4f5382",'font-size':'16px'}),
                                                    html.Hr(),  
                                                    dcc.Loading(                                          
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-cash-financial-literacy-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                        html.Div(id='value_predit_rl_test_output',style={"color" : "#4f5382",'font-size':'16px'}),                                                        
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),  # Corregido aquí
                                                    html.Br(),
                                                    dcc.Loading( 
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                        html.Div(id='r2_test_rl_output',style={"color" : "#4f5382",'font-size':'16px'}), 
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),
                                                    html.Br(),   
                                                    html.H1("Lasso Prediction",style={"color" : "#4f5382",'font-size':'16px'}),
                                                    html.Hr(),
                                                    dcc.Loading(                                            
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-cash-financial-literacy-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                        html.Div(id='value_predit_l_test_output',style={"color" : "#4f5382",'font-size':'16px'}),                                                        
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),  # Corregido aquí
                                                    html.Br(),
                                                    dcc.Loading( 
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                        html.Div(id='r2_test_l_output',style={"color" : "#4f5382",'font-size':'16px'}), 
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),
                                                    html.Br(),   
                                                    html.H1("Ridge Prediction",style={"color" : "#4f5382",'font-size':'16px'}),
                                                    html.Hr(),
                                                    dcc.Loading(                                            
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-cash-financial-literacy-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                        html.Div(id='value_predit_r_test_output',style={"color" : "#4f5382",'font-size':'16px'}),                                                        
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),  # Corregido aquí
                                                    html.Br(),
                                                    dcc.Loading(  
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                        html.Div(id='r2_test_r_output',style={"color" : "#4f5382",'font-size':'16px'}), 
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),
                                                    html.Br(),   
                                                    html.H1("Support Vector Machines - RBF Prediction",style={"color" : "#4f5382",'font-size':'16px'}),
                                                    html.Hr(),
                                                    dcc.Loading(                                            
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-cash-financial-literacy-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                        html.Div(id='value_predit_svm_test_output',style={"color" : "#4f5382",'font-size':'16px'}),                                                        
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),  # Corregido aquí
                                                    html.Br(), 
                                                    dcc.Loading( 
                                                    html.Div([
                                                        html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio.png", alt="external-evaluation-company-plan-inipagistudio-lineal-color-inipagistudio", style={'width':'40px'}),
                                                        html.Div(id='r2_test_svm_output',style={"color" : "#4f5382",'font-size':'16px'}), 
                                                    ], style= {'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})),   
                                                    html.Br(),                                         
                                                    ],
                                                    align="center",                                                    
                                                ),                                                                                       
                                                ]),
                                        ]
                                    ),
                                ),
                                sm=4,
                            ),
                        ],
                        align="center",
                    ),
                ]
            ),
            style={
                'overflowY': 'auto',      # Permite desplazamiento si es necesario
                'height': '80vh',         # Ajusta la altura del CardBody para ocupar más de la pantalla
                'padding-bottom': '0px'         # Ajusta el padding para mejor presentación
                }
        ),                         
            ],style={
        'padding-bottom': '0',    # Reduce padding si no es necesario
        'height': '80vh', # Asegura que todo el contenido ocupe la altura completa
    }
)


##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                                                              CALLBACKS                                                                                         #
#                                                                                                                                                                                #
##################################################################################################################################################################################

# Callback para (KNN)

@callback(
    Output(component_id='value_predit_knn_test_output', component_property='children'),
    Output(component_id='r2_test_output', component_property='children'),
    Input(component_id="slct_far_sea", component_property='value')
)
def update_dataframe(valor_slct_far_sea):
    
    # Filtrar los datos según la selección del dropdown
    df_filtered = df_r[df_r['ocean_proximity'] == valor_slct_far_sea]

    # Llamar a la función modeloknn con el DataFrame filtrado
    pred_price, r2_test = modeloknn(df_filtered)

    
    # Retornar el precio predicho y el R²
    return f"Predicted Value: {pred_price:,.0f} USD", f"R2 Score: {r2_test*100:.2f}%"

#Callback para Linear Regression ##############################################################

@callback(
    Output(component_id='value_predit_rl_test_output', component_property='children'),
    Output(component_id='r2_test_rl_output', component_property='children'),
    Input(component_id="slct_far_sea", component_property='value')
)
def update_dataframe(valor_slct_far_sea):
    
    # Filtrar los datos según la selección del dropdown
    df_filtered = df_r[df_r['ocean_proximity'] == valor_slct_far_sea]

    # Llamar a la función modeloknn con el DataFrame filtrado
    pred_price, r2_test =modelorl(df_filtered)

    
    # Retornar el precio predicho y el R²
    return f"Predicted Value: {pred_price:,.0f} USD", f"R2 Score: {r2_test*100:.2f}%"



#Call Back para lasso ################################################################################ 
@callback(
    Output(component_id='value_predit_l_test_output', component_property='children'),
    Output(component_id='r2_test_l_output', component_property='children'),
    Input(component_id="slct_far_sea", component_property='value')
)
def update_dataframe(valor_slct_far_sea):
   
    # Filtrar los datos según la selección del dropdown
    df_filtered = df_r[df_r['ocean_proximity'] == valor_slct_far_sea]

    # Llamar a la función modeloknn con el DataFrame filtrado
    pred_price, r2_test = modelolasso(df_filtered)

    
    # Retornar el precio predicho y el R²
    return f"Predicted Value: {pred_price:,.0f} USD", f"R2 Score: {r2_test*100:.2f}%"

#call Back para Ridge ################################################################################ 

@callback(
    Output(component_id='value_predit_r_test_output', component_property='children'),
    Output(component_id='r2_test_r_output', component_property='children'),
    Input(component_id="slct_far_sea", component_property='value')
)
def update_dataframe(valor_slct_far_sea):
    
    # Filtrar los datos según la selección del dropdown
    df_filtered = df_r[df_r['ocean_proximity'] == valor_slct_far_sea]

    # Llamar a la función modeloknn con el DataFrame filtrado
    pred_price, r2_test = modeloridge(df_filtered)

    
    # Retornar el precio predicho y el R²
    return f"Predicted Value: {pred_price:,.0f} USD", f"R2 Score: {r2_test*100:.2f}%"


#CallBack para SVM  ################################################################################ 

@callback(
    Output(component_id='value_predit_svm_test_output', component_property='children'),
    Output(component_id='r2_test_svm_output', component_property='children'),
    Input(component_id="slct_far_sea", component_property='value')
)
def update_dataframe(valor_slct_far_sea):
    
    # Filtrar los datos según la selección del dropdown
    df_filtered = df_r[df_r['ocean_proximity'] == valor_slct_far_sea]

    # Llamar a la función modeloknn con el DataFrame filtrado
    pred_price, r2_test = modeloSVM(df_filtered)

    
    # Retornar el precio predicho y el R²
    return f"Predicted Value: {pred_price:,.0f} USD", f"R2 Score: {r2_test*100:.2f}%"




#Callback para estimaciones puntuales  ################################################################################ 

@callback(
    Output(component_id='rooms_p_output', component_property='children'),
    Output(component_id='bedrooms_p_output', component_property='children'),
    Output(component_id='median_inc_p_output', component_property='children'),
    Input(component_id="slct_far_sea", component_property='value')
)
def update_dataframe(valor_slct_far_sea):

    # Llamar a la función calculos
    rooms_p, bedrooms_p, median_inc_p = number_spaces(df_r, valor_slct_far_sea)

    
    # Retornar el precio predicho y el R²
    return f" Average number of Rooms: {rooms_p}", f" Average number of Bedrooms: {bedrooms_p}", f"Average annual income: ${median_inc_p:,.0f} USD"



if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(debug=True, host='0.0.0.0', port=9000) # <- To Dockerize the Dash