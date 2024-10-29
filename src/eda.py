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

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf



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
df = pd.read_csv("data_2017_2022sinnan.csv", sep=";")
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
df_t = df_t.drop(columns=["HR", "RF", "P", "RS", "AT", "WS", "O3", "SO2", "PM2.5", "WD"])
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
orden_horas = [f"{i:02d}:00" for i in range(24)]
horas_a_numero = {f"{i:02d}:00": i for i in range(24)} # Crear un diccionario para mapear números de hora a nombres
df_t['hora'] = df_t['fecha'].dt.hour.map(lambda x: f"{x:02d}:00")
df_t['hora_num'] = df_t['hora'].map(horas_a_numero)


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
                            html.H2(
                                "Análisis Descriptivo de las Mediciones de PM10 Registradas", 
                                style={"color": "#4f5382", 'font-size': '18px'}
                            ), 
                            html.Hr(),                          
                            html.Br(),
                            html.H2(
                                "Selecciona el año de tu Preferencia para Analizar los Valores de PM10 Registrados", 
                                style={"color": "#4f5382", 'font-size': '16px'}
                            ),
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
                                   dcc.Graph(id="pm10_year_bar",figure={}),# id de gráfico de barra por años  
                                ])
                            ),
                            html.Br(),
                            dcc.Loading(
                                html.Div([
                                   dcc.Graph(id="pm10_year_bar_day",figure={}),# id de gráfico de barra por años  
                                ])
                            ),
                            html.Br(),
                            dcc.Loading(
                                html.Div([
                                   dcc.Graph(id="pm10_year_bar_hour",figure={}),# id de gráfico de barra por años  
                                ])
                            ),
                            html.Br(),
                            dcc.Loading(
                                html.Div([
                                   dcc.Graph(id="pm10_distribution",figure={}),# id de gráfico de distribución PM10
                                ])
                            ),
                            html.Br(),
                            html.Br(),
                            html.Br(),
                        ])
                    ),
                    sm=6,
                ),
                dbc.Col(
                    html.Div([
                       html.H1(
                           "Comparaciones de tamaño para partículas de PM",
                           style={'width': '100%', 'height': 'auto', 'color': "#4f5382", 'font-size': '18px'}
                        ),
                        html.Img(
                            src=f'data:image/png;base64,{imagen_codificada}',
                            style={'width': '60%', 'height': 'auto', 'display': 'block', 'margin': '0 auto'}
                        ),
                        dbc.Card(
                            dbc.CardBody([
                                html.H1("Análisis de Series Temporales", style={"color": "#4f5382", 'font-size': '18px'}),
                                html.Hr(),
                            dcc.Loading(
                                html.Div([
                                   dcc.Graph(id="pm10_serie_time",figure={}),# id de gráfico de serie de tiempo
                                ])
                            ), 
                            html.Br(),
                            html.H5(
                                "Seleccione el periodo de su preferencia para visualizar los gráficos de Autocorrelación, Autocorrelación Parcial y  descomposición de la serie temporal de la concentración de PM10:", 
                                style={"color": "#4f5382", 'font-size': '16px'}
                            ),
                            dcc.Dropdown(
                                id="slct_period",
                                options=[
                                    {"label": "Diario", "value": 24},
                                    {"label": "Semanal", "value": 168},
                                    {"label": "Mensual", "value": 720},
                                    {"label": "Trimestral", "value": 2160},
                                    {"label": "Semestral", "value": 4320},
                                    {"label": "Anual", "value": 8640},
                                ],
                                multi=False,
                                value=24,
                                style={'width': "100%", 'font-size': '15px', 'border-radius': '30px'}
                            ),                            
                            html.Br(),
                                dcc.Loading(
                                html.Div([
                                   dcc.Graph(id="ACF_g",figure={}),# id de gráfico de descomposición de serie de tiempo
                                ])
                            ), 
                                dcc.Loading(
                                html.Div([
                                   dcc.Graph(id="PACF_g",figure={}),# id de gráfico de descomposición de serie de tiempo
                                ])
                            ),  
                            dcc.Loading(
                                html.Div([
                                   dcc.Graph(id="pm10_des_serie_time",figure={}),# id de gráfico de descomposición de serie de tiempo
                                ])
                            ),                      
                            ]),
                        ),
                    ], className="d-flex flex-column align-items-center"),
                    sm=6,
                ),
            ]),
        ]),
        style={
            'overflowY': 'auto',
            'height': '100vh',
            'padding-bottom': '0px'
        }
    )
], style={
    'padding-bottom': '0',
    'height': '90vh',
})


#################################################################################################################################################
#                                                                                                                                                #                                
#                                                                              CALLBACKS                                                         #                                
#                                                                                                                                                #                                
#################################################################################################################################################

# Llamada para gráfico de Boxplot por mes ######################################################################################
@callback(
    Output(component_id='pm10_year_bar', component_property='figure'),  
    Input(component_id="slct_year", component_property='value')  
)
def make_graph_box_month(slct_year):
  
    colores_meses = ['#30143F', '#432c4d', '#5a3b59', '#734764', '#8f615c', 
                     '#a86e4f', '#c67e3f', '#d98e31', '#e69c24', '#f2aa1c', 
                     '#f5b512', '#f8b902']
    dfb = df_t.copy()
    dfb = dfb[dfb["año"] == slct_year]  
    
    # Crear una figura para el boxplot
    figbox = go.Figure()
    
    # Añadir trazas para cada mes con colores distintos
    for mes, color in zip(range(1, 13), colores_meses):  
        df_subset = dfb[dfb['mes_num'] == mes]  
        figbox.add_trace(go.Box(
            y=df_subset['PM10'],
            name=orden_meses[mes - 1], 
            marker_color=color
        ))
    
    # Actualizar el diseño del gráfico
    figbox.update_layout(
        title="Distribución de PM10 Mensual",
        xaxis_title="Día",
        yaxis_title="PM10 (µg/m³)",
        title_x=0.5,  
        xaxis=dict(
            tickmode='array',
            tickvals=orden_meses,  
            ticktext=orden_meses,  
            categoryorder='array',  
            categoryarray=orden_meses 
        )
    )

    return figbox

# Llamada para gráfico de Boxplot por día ######################################################################################
@callback(
    Output(component_id='pm10_year_bar_day', component_property='figure'), 
    Input(component_id="slct_year", component_property='value')  
)
def make_graph_box_day(slct_year):

    colores_dias = ['#30143F', '#432c4d', '#5a3b59', '#734764', '#8f615c', 
                    '#a86e4f', '#c67e3f', '#d98e31', '#e69c24', '#f2aa1c', 
                    '#f5b512', '#f8b902']  
    dfb = df_t.copy()
    dfb = dfb[dfb["año"] == slct_year]  
    
    # Crear una figura para el boxplot
    figbox = go.Figure()
    
    # Añadir trazas para cada día de la semana con colores distintos
    for dia_num, color in zip(range(7), colores_dias):  
        df_subset = dfb[dfb['dia_num'] == dia_num]  
        figbox.add_trace(go.Box(
            y=df_subset['PM10'],
            name=orden_dias[dia_num],  
            marker_color=color
        ))
    
    # Actualizar el diseño del gráfico
    figbox.update_layout(
        title="Distribución de PM10 Diario",
        xaxis_title="Día de la Semana",
        yaxis_title="PM10 (µg/m³)",
        title_x=0.5,  
        xaxis=dict(
            tickmode='array',
            tickvals=orden_dias,  
            ticktext=orden_dias,  
            categoryorder='array',  
            categoryarray=orden_dias  
        )
    )

    return figbox


# Llamada para gráfico de Boxplot por hora ######################################################################################
@callback(
    Output(component_id='pm10_year_bar_hour', component_property='figure'),  
    Input(component_id="slct_year", component_property='value')  
)
def make_graph_box_hour(slct_year):
    colores_horas = ['#30143F', '#432c4d', '#5a3b59', '#734764', '#8f615c', '#a86e4f', '#c67e3f', 
                 '#d98e31', '#e69c24', '#f2aa1c', '#f5b512', '#f8b902', '#f8d024', '#f4c433', 
                 '#f0b838', '#d8a73d', '#c09d42', '#a9914f', '#8c8e5a', '#6d8a65', '#51776c', 
                 '#356d6f', '#1e616c', '#115f6a', '#006f6f']
    dfb = df_t.copy()
    dfb = dfb[dfb["año"] == slct_year]  
    
    # Crear una figura para el boxplot
    figbox = go.Figure()
    
    for hora, color in zip(orden_horas, colores_horas):
        df_subset = dfb[dfb['hora'] == hora]
        figbox.add_trace(go.Box(
            y=df_subset['PM10'],
            name=hora,
            marker_color=color
        ))

    # Actualizar el diseño del gráfico
    figbox.update_layout(
        title="Distribución de PM10 por Hora del Día",
        xaxis_title="Hora del Día",
        yaxis_title="PM10 (µg/m³)",
        title_x=0.5,  # Centrar el título
        xaxis=dict(
            tickmode='array',
            tickvals=orden_horas,
            ticktext=orden_horas,  
            categoryorder='array',  # Ordenar las horas en el orden especificado
            categoryarray=orden_horas  # Especificar el orden de las horas
        )
    )


    return figbox



# Llamada para gráfico de distribución #########################################################################
@callback(
    Output(component_id='pm10_distribution', component_property='figure'),  
    Input(component_id="slct_year", component_property='value')
)
def make_graph_distribution(slct_year):
    
    dfb = df_t.copy()
    dfb = dfb[dfb["año"] == slct_year]
    dfb = dfb['PM10']  
    
    fig_density = px.histogram(dfb, x="PM10", marginal="box", nbins=50, title="Gráfico de Densidad de PM10",
                            labels={'PM10': 'PM10 (µg/m³)'}, color_discrete_sequence=['#8C7C6D'])


    fig_density.update_traces(
        marker=dict(color='#8C7C6D'),  # Ajustar la opacidad de las barras
        selector=dict(type='histogram')
    )

    # Ajustar el diseño del gráfico de densidad
    fig_density.update_layout(
        xaxis_title="PM10 (µg/m³)",
        yaxis_title="Frecuencia",
        showlegend=False,
        template='plotly_white' 
    )
    return fig_density


# Llamada para gráfico de serie de tiempo #########################################################################
@callback(
    Output(component_id='pm10_serie_time', component_property='figure'),  
    Input(component_id="slct_year", component_property='value')
)
def make_graph_st(slct_year):
    
    dfb = df_t.copy()
    dfb = dfb[dfb["año"] == slct_year]  
    
    figst = go.Figure()
    
    figst.add_trace(go.Scatter(
        x=dfb['fecha'],
        y=dfb['PM10'],
        mode='lines',  # Mostrar la línea
        line=dict(color='#8C7C6D', width=2),  # Cambiar el color de la línea y el ancho
        name='PM10 (µg/m³)'
    ))

    # Actualizar el diseño del gráfico
    figst.update_layout(
        title="Distribución de PM10 a lo largo del tiempo",
        xaxis_title="Fecha",
        yaxis_title="PM10 (µg/m³)",
        xaxis=dict(tickangle=45, tickformat='%Y-%m-%d'),  # Rotar etiquetas del eje X y formato de fecha
        yaxis=dict(tickformat='.1f'),  # Formato de los ticks del eje Y (un decimal)
        template='plotly_white',  # Fondo blanco 
        title_x=0.5,  # Centrar el título
        title_font=dict(size=16)  # Ajusta el tamaño de la fuente del título
    )
    return figst


# Llamada para gráfico de ACF ######################################################################################
@callback(
    Output(component_id='ACF_g', component_property='figure'),  
    Input(component_id="slct_year", component_property='value'),
    Input(component_id="slct_period", component_property='value')
)
def plot_acf_dynamic(slct_year, slct_period):

    dfb = df_t.copy()
    dfb = dfb[dfb["año"] == slct_year]
    dfb.set_index('fecha', inplace=True)
    pm10_series = dfb['PM10']

    # Calcular la ACF con los lags especificados
    acf_values = acf(pm10_series, nlags=slct_period, fft=False)

   # Crear un gráfico interactivo con Plotly
    fig = go.Figure()
    
     # Añadir las barras delgadas
    fig.add_trace(go.Bar(
        x=np.arange(len(acf_values)), 
        y=acf_values, 
        marker=dict(color='#41A8BF', line=dict(width=1)),  # Barras delgadas
        width=0.06,  # Grosor de las barras
        name='ACF',
        showlegend=False
    ))

    # Añadir las barras delgadas
    fig.add_trace(go.Scatter(
        x=np.arange(len(acf_values)), 
        y=acf_values, 
        mode='markers',  # Barras con puntos en la parte superior
        line=dict(color='#41A8BF', width=80),  # Líneas delgadas
        marker=dict(size=6, color='#41A8BF'),  # Tamaño y color de los puntos
        name='ACF'
    ))

    # Calcular los límites de significancia
    conf_interval = 1.96 / np.sqrt(len(pm10_series))

    # Añadir el área de significancia
    fig.add_trace(go.Scatter(
        x=np.arange(len(acf_values)), 
        y=[conf_interval] * len(acf_values),
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(acf_values)), 
        y=[-conf_interval] * len(acf_values),
        fill='tonexty',  # Rellenar el área entre los dos valores
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(0, 128, 255, 0.2)',  # Color del área azul
        name='Conf. Interval'
    ))

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title="Gráfica de Autocorrelación (ACF)",
        xaxis_title="Lags",
        yaxis_title="ACF Value",
        template="plotly_white",
        showlegend=False
    )

    return fig


# Llamada para gráfico de PACF ######################################################################################

@callback(
    Output(component_id='PACF_g', component_property='figure'),  
    Input(component_id="slct_year", component_property='value'),
    Input(component_id="slct_period", component_property='value')
)   
def plot_pacf_dynamic(slct_year, slct_period):

    dfb = df_t.copy()
    dfb = dfb[dfb["año"] == slct_year]
    dfb.set_index('fecha', inplace=True)
    pm10_series = dfb['PM10']

    # Calcular la ACF con los lags especificados
    pacf_values = pacf(pm10_series, nlags=slct_period)

   # Crear un gráfico interactivo con Plotly
    fig = go.Figure()
    
     # Añadir las barras delgadas
    fig.add_trace(go.Bar(
        x=np.arange(len(pacf_values)), 
        y=pacf_values, 
        marker=dict(color='#8C7C6D', line=dict(width=1)),  # Barras delgadas
        width=0.06,  # Grosor de las barras
        name='ACF',
        showlegend=False
    ))

    # Añadir las barras delgadas
    fig.add_trace(go.Scatter(
        x=np.arange(len(pacf_values)), 
        y=pacf_values, 
        mode='markers',  # Barras con puntos en la parte superior
        line=dict(color='#8C7C6D', width=80),  # Líneas delgadas
        marker=dict(size=6, color='#8C7C6D'),  # Tamaño y color de los puntos
        name='ACF'
    ))

    # Calcular los límites de significancia
    conf_interval = 1.96 / np.sqrt(len(pm10_series))

    # Añadir el área de significancia
    fig.add_trace(go.Scatter(
        x=np.arange(len(pacf_values)), 
        y=[conf_interval] * len(pacf_values),
        fill=None,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(pacf_values)), 
        y=[-conf_interval] * len(pacf_values),
        fill='tonexty',  # Rellenar el área entre los dos valores
        mode='lines',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(0, 128, 255, 0.2)',  # Color del área azul
        name='Conf. Interval'
    ))

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title="Gráfica de Autocorrelación Parical (PACF)",
        xaxis_title="Lags",
        yaxis_title="PACF Value",
        template="plotly_white",
        showlegend=False
    )

    return fig

# Llamada para gráfico desomposición de serie de tiempo ######################################################################################
@callback(
    Output(component_id='pm10_des_serie_time', component_property='figure'),  
    Input(component_id="slct_year", component_property='value'),
    Input(component_id="slct_period", component_property='value')
)
def make_decomposition_graph(slct_year, slct_period):
    dfb = df_t.copy()
    dfb = dfb[dfb["año"] == slct_year]
    periodo = slct_period  # Cambia esto según la estacionalidad esperada
    decomposition = seasonal_decompose(dfb['PM10'], model='additive', period=periodo)

    # Crear subplots con layout
    fig = make_subplots(
        rows=4, 
        cols=1, 
        subplot_titles=("Observado", "Tendencia", "Estacionalidad", "Residual"),
        vertical_spacing=0.1  # Espacio vertical entre subplots
    )

    fig.add_trace(go.Scatter(
        x=dfb['fecha'],
        y=dfb['PM10'],
        mode='lines',
        name='Observado',
        showlegend=False,
        line=dict(color='#8C7C6D', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dfb['fecha'],
        y=decomposition.trend,
        mode='lines',
        name='Tendencia',
        line=dict(color='#A66946', width=2),
        showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dfb['fecha'],
        y=decomposition.seasonal,
        mode='lines',
        name='Estacionalidad',
        line=dict(color='#41A8BF', width=2),
        showlegend=False 
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=dfb['fecha'],
        y=decomposition.resid,
        mode='lines',
        name='Residual',
        line=dict(color='#F2B077', width=2),
        showlegend=False 
    ), row=4, col=1)

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title="Descomposición Estacional de PM10",
        template='plotly_white',
        title_x=0.5,  # Centrar el título
        height=800  # Ajustar la altura del gráfico
    )

    return fig

   
if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(debug=True, host='0.0.0.0', port=9000) # <- To Dockerize the Dash