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


#################################################################################################################################################
#                                                                                                                                                #                               
#                                                              TRATAMIENTO DE DATOS                                                              #                               
#                                                                                                                                                #                                
#################################################################################################################################################

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

# Agregar columnas a dataframe de año
df_t['año'] = df_t['fecha'].dt.year  # Extraer el año de la columna 'fecha'

# Agregar columnas a dataframe mes número y mes nombre 

df_t['mes_num'] = df_t['fecha'].dt.month  
orden_meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
               "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
nombres_meses = {i+1: mes for i, mes in enumerate(orden_meses)}
df_t['mes'] = df_t['mes_num'].map(nombres_meses)  


# Agregar columnas a dataframe  día número y día nombre 
df_t['dia_num'] = df_t['fecha'].dt.dayofweek   # Extraer el año de la columna 'día
orden_dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
nombres_dias = {i: dia for i, dia in enumerate(orden_dias)}
df_t['dia'] = df_t['dia_num'].map(nombres_dias)



# Agregar columnas a dataframe hora
df_t['hora'] = df_t['fecha'].dt.hour  # Extraer el año de la columna 'mes'
df_t['hora'] = df_t['fecha'].dt.strftime('%H:%M')  # Formato HH:MM

#print(df_t)







#################################################################################################################################################
#                                                                                                                                               #                               
#                                               DEFINICIÓN DE FUNCIONES MODELOS ML                                                               #                              
#                                                                                                                                                #                           
#################################################################################################################################################






 






 


###############################################################################################################################################                                                                                                                                                                                                             CREAR APLICACIÓN DASH                                                                                          
                                                                                                                                                                             
#################################################################################################################################################

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
    path='/eda',
    title="How neural network learns",
    description="How neural network learns",
    image='hogar.png'
)



#############################################################################################################################################
#                                                                                                                                                #                                
#                                                          CONFIGURACIÓN DE GRÁFICOS                                                             #                              
#                                                                                                                                                #                                
###############################################################################################################################################

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
Home = 'assets/pm10_scale_graphic2.png'
with open(Home, 'rb') as image_file:
    imagen_codificada = base64.b64encode(image_file.read()).decode('utf-8')

# https://www.epa.gov/pm-pollution/particulate-matter-pm-basics 




##################################################################################################################################################################################
#                                                                                                                                                                                #
#                                                                  CONTENIDO DE LAYOUT                                                                                           #
#                                                                                                                                                                                #
##################################################################################################################################################################################


layout_eda = html.Div([
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody([
                            html.Br(),
                            html.Br(),
                            html.H2(
                                "Selecciona tu Opción de Preferencia para Analizar los Valores de PM10 Registrados", 
                                style={"color": "#4f5382", 'font-size': '16px'}
                            ),
                            html.Br(),
                            html.H5("Registro de PM10 durante el(los) Año(s):", style={"color": "#4f5382", 'font-size': '16px'}),
                            dcc.Dropdown(
                                id="slct_year",
                                options=[
                                    {"label": "2017", "value": 2017},
                                    {"label": "2018", "value": 2018},
                                    {"label": "2019", "value": 2019},
                                    {"label": "2020", "value": 2020},
                                    {"label": "2021", "value": 2021},
                                    {"label": "2022", "value": 2022},
                                ],
                                multi= False,
                                value=2017, 
                                style={'width': "100%", 'font-size': '15px', 'border-radius': '30px'}
                            ),
                            html.Br(),
                            dcc.Loading(
                                html.Div([
                                    dcc.Graph(figure=mapa, style={
                                        'width': '100%',
                                        'height': '50vh',
                                        'display': 'flex',
                                        'align-items': 'center',
                                        'justify-content': 'center',
                                    })
                                ])
                            ),
                            html.Br(), 
                            dcc.Loading(
                                html.Div([
                                   dcc.Graph(id="pm10_year_bar"),# id de gráfico de barra por años  
                                ])
                            ),
                            html.Br(),
                            html.H5(
                                "Seleccione el periodo que prefiera para visualizar la descomposición de la serie temporal de la concentración de PM10:", 
                                style={"color": "#4f5382", 'font-size': '16px'}
                            ),
                            dcc.Dropdown(
                                id="slct_month",
                                options=[
                                    {"label": "Diario", "value": "D"},
                                    {"label": "Semanal", "value": "W"},
                                    {"label": "Mensual", "value": "M"},
                                    {"label": "Trimestral", "value": "Q"},
                                    {"label": "Semestral", "value": "2Q"},
                                    {"label": "Anual", "value": "A"},
                                ],
                                multi=False,
                                value="D",
                                style={'width': "100%", 'font-size': '15px', 'border-radius': '30px'}
                            ),
                            html.Br(),
                            dcc.Loading(
                                html.Div([
                                    html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-tower-spain-inipagistudio-lineal-color-inipagistudio.png", 
                                             style={'width': '40px'}),
                                    html.Div(id='rooms_p_output', style={"color": "#4f5382", 'font-size': '16px'}),
                                ], style={'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})
                            ),
                            html.Br(),
                            dcc.Loading(
                                html.Div([
                                    html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-bed-house-keeping-inipagistudio-lineal-color-inipagistudio.png", 
                                             style={'width': '40px'}),
                                    html.Div(id='bedrooms_p_output', style={"color": "#4f5382", 'font-size': '16px'}),
                                ], style={'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})
                            ),
                            html.Br(),
                            dcc.Loading(
                                html.Div([
                                    html.Img(src="https://img.icons8.com/external-inipagistudio-lineal-color-inipagistudio/64/external-money-management-financial-literacy-inipagistudio-lineal-color-inipagistudio.png", 
                                             style={'width': '40px'}),
                                    html.Div(id='median_inc_p_output', style={"color": "#4f5382", 'font-size': '16px'}),
                                ], style={'display': 'flex', 'align-items': 'flex-end', 'gap': '10px'})
                            ),
                        ])
                    ),
                    sm=7,
                ),
                dbc.Col(
                    html.Div([
                        html.H1(
                            "Comparaciones de tamaño para partículas de PM",
                            style={'width': '100%', 'height': 'auto', 'color': "#4f5382", 'font-size': '18px'}
                        ),
                        html.Img(
                            src=f'data:image/png;base64,{imagen_codificada}',
                            style={'width': '80%', 'height': 'auto', 'display': 'block', 'margin': '0 auto'}
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                html.H1("Otra información o gráfica de interés", style={"color": "#4f5382", 'font-size': '18px'}),
                                html.Hr(),
                                dbc.Row([]),
                            ])
                        ),
                    ], className="d-flex flex-column align-items-center"),
                    sm=4,
                ),
            ]),
        ]),
        style={
            'overflowY': 'auto',
            'height': '80vh',
            'padding-bottom': '0px'
        }
    )
], style={
    'padding-bottom': '0',
    'height': '80vh',
})




#################################################################################################################################################
#                                                                                                                                                #                                
#                                                                              CALLBACKS                                                         #                                
#                                                                                                                                                #                                
#################################################################################################################################################

# Llamada para gráfico de Boxplot 
@app.callback(
    Output(component_id='pm10_year_bar', component_property='figure'),  # Definición de salida del gráfico de barra
    Input(component_id="slct_year", component_property='value')  # Definición de input: año(s) a seleccionar
)
def make_graph_box(slct_year):
    print("Ingreso")
    colores_meses = ['#30143F', '#432c4d', '#5a3b59', '#734764', '#8f615c', 
                     '#a86e4f', '#c67e3f', '#d98e31', '#e69c24', '#f2aa1c', 
                     '#f5b512', '#f8b902']
    dfb = df_t.copy()
    dfb = dfb[dfb["año"] == slct_year]  
    
    # Crear una figura para el boxplot
    figbox = go.Figure()
    
    # Añadir trazas para cada mes con colores distintos
    for mes, color in zip(range(1, 13), colores_meses):  # Usar números del mes (1-12)
        df_subset = dfb[dfb['mes_num'] == mes]  # Filtrar por el número de mes
        figbox.add_trace(go.Box(
            y=df_subset['PM10'],
            name=orden_meses[mes - 1],  # Usar la lista de nombres de meses
            marker_color=color
        ))
    
    # Actualizar el diseño del gráfico
    figbox.update_layout(
        title="Distribución de PM10 Mensual",
        xaxis_title="Mes",
        yaxis_title="PM10 (µg/m³)",
        title_x=0.5,  
        xaxis=dict(
            tickmode='array',
            tickvals=orden_meses,  
            ticktext=orden_meses,  
            categoryorder='array',  # Ordenar los meses en el orden especificado
            categoryarray=orden_meses  # Especificar el orden de los meses
        )
    )

    return figbox
    
if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(debug=True, host='0.0.0.0', port=9000) # <- To Dockerize the Dash