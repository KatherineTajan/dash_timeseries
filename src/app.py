#Librerias para no borrar 

import dash 
import math
from dash import Dash, Input, Output,  State, callback, dash_table, html, dcc 
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

from dash_bootstrap_components import Navbar
from sklearn.model_selection import train_test_split, GridSearchCV, KFold,validation_curve
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,LabelEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay,confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

# Importa el layout desde book_now.py
from book_now import layout_book
from eda import layout_eda
#import app 

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import base64


#########################################################################################################################################################################################

# Crear aplicación Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.UNITED, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    external_scripts=['https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML']
)

# Códificación de la imagen
home1 = 'assets/download.png'
with open(home1, 'rb') as image_file:
    imagen_codificada = base64.b64encode(image_file.read()).decode('utf-8')

# Layout de la página principal sin imagen de fondo
home_layout = html.Div(
)

data_upload_layout = html.Div(
    children=[html.H1(children="This is our upload page")]
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.I(className='bi bi-globe-americas', style={'font-size': '30px', 'color': '#262626'}),  # Color agregado
                            dbc.NavbarBrand('Predicción de Material Particulado de 10μm (PM10)', className='ms-2', style={'font-size': '18px', 'color': '#262626'})
                        ],
                        width={'size': 'auto'}
                    )
                ],
                align='center'
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Nav(
                                [
                                    dbc.NavItem(dbc.NavLink("Inicio", href="/", style={'font-size': '18px', 'color': '#262626'})),  
                                    dbc.NavItem(dbc.NavLink("EDA", href="/eda", style={'font-size': '18px', 'color': '#262626'})),  
                                    dbc.NavItem(dbc.NavLink("Modelos Predictivos", href="/book_now", style={'font-size': '18px', 'color': '#262626'})),  
                                ],
                                navbar=True
                            )
                        ],
                        width={'size': 'auto'}
                    )
                ],
                align='center'
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Collapse(
                                dbc.Nav(
                                    [
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                html.I(className='bi bi-github', style={'font-size': '25px', 'color': '#262626'}),
                                                href="https://katherinetajan.github.io/proyecto_ST/",
                                                external_link=True
                                            )
                                        ),
                                        dbc.NavItem(
                                            dbc.NavLink(
                                                html.I(className='bi bi-database-down', style={'font-size': '25px', 'color': '#262626'}),
                                                href="https://pdf.sciencedirectassets.com/271090/1-s2.0-S0360544221X00266/1-s2.0-S0360544221030619/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEJP%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJIMEYCIQCa8ZdJDnEpBW%2FbeZpIdpWu8WOmSU1JhplQA88SVzmpEwIhAIgA9BzPniSW7z5kxVuT7KdLQuZ0BC%2FGUK75HFAkYPcJKrsFCPz%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQBRoMMDU5MDAzNTQ2ODY1IgxluHjOTFUV015pbsgqjwWpMTJ8oOFv1DicDklmBFTFlOhtlwilf8g8afFfv4QBVkadFsjmU%2B3C1RUvDGgt2KjNPoKaVfI15rth2q5ZGkA4vLJP71spPQHEBEV4ka8XGZn0K6N1ksw6OoEe950IYK%2BE48VckkMNkHMDyDaex0Czv8e5EaLTIC4Ud0u3DZDfz%2B%2FKiFzLTx8UAHgNPIMcZmTq74aluq06UpDcEy0Tpth%2FbWGUKIx19wW8ENsrjU%2BnP9QYfNOM0YCvlJzAWSKYBp6LX7nmVuBZJqTjsXwzgQyhlPkktY%2B%2BcqvWlQmsviw55fV7wWkwVFXtUMbD0oBknsYyvUTAHJrd1lRAcYcX5YfEPmOaUabfxwFr4bjjZC3CcsUv822BcP%2FeFnihguVZ8gCqX7qH5sF9NkFHzv4m0niJ7%2BOwZymDgrGG%2Ft124DFaR2IZH29RTz7DnR7JgWx%2B%2BreSyePmXJC%2Bu4CjczxOiwnPi4LTHZZ7QzdeT2X2b1cGOTKFuJ9T8bewpf1Ro3dTHNoyPnHhnz6lXj31L0u6iKC27tRjohIIaD9zo8a8%2BZt7GSJUyDjvWCUO6Y3hvR5tddhfRSBHoQsTMMInrUPCWXwI1kURyVrzjXW5Jtv30am29wpLPHZdhOXJFk2ngu7H%2FXyq%2B73tVXwqLv7xdEsZbr6MpEMxrbx836pqr6AFcNaWZH9urfzwtlN6MTsZEs1CpkU4Dlarbi9iWIwWnJKxLb6AFfPtSMapkPzT2CDxZ7B9pUjMesP0aPDZ1quR9a1BfaWd1SrOxlOXp1hyKRtmUWNGtUqnj82%2BPB2prgJ8lCNu%2BnEHG6A1qFHIfx4RlGriuQNHJDBrmjmroRcCc%2BVra6m5iOGwAOSZBtb0zU4NRqJCMJGl8bgGOrABod14OIqXykq3IqMRoR6pVAkxdQAbSUTaBKtLNx2pNwM3c2nC%2BLMlZhffg4hL7fDuBRsLnfiXlRR8WhL5SErjbY3XPh8h1ikhmKODjdJseuPCyCUahNWVLXnCWyOvAW1xlkU904o9q41Fz3Btr%2Bq7ai6SWhmWASMLkB8ENMLP%2FFPWLqiBY4g1vIjhBy2jRKe%2BVr7OpG7CMMnPIghvSzh%2FGbKpMCoTrLV2bj2D8DBTBdI%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20241026T025044Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY4RIIDM6T%2F20241026%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=55462a4dd228add2e0dec424ad4347865f40dab89a4858094bf4bfa95b081128&hash=f404b4cf8e9b544f994273324cdc98ad65e972d5a6d4039e708eb2aa1dbc0749&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0360544221030619&tid=spdf-3e3f98e5-fe6f-4e4f-837c-6b145b35c4cd&sid=de83f2ea2bf9544d671a6a375d259fe8367egxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=050f5f0358585452065e&rr=8d8723f97a3ba55e&cc=co",
                                                external_link=True
                                            )
                                        )
                                    ],
                                    navbar=True
                                ),
                                id="navbar-collapse",
                                is_open=False,
                                navbar=True
                            )
                        ],
                        width={'size': 'auto'}
                    )
                ],
                align='center'
            ),
        ],
        fluid=True
    ),
    color="#8C7C6D",
    dark=True,
    className="mb-2",
)

# Layout principal con imagen de fondo configurada adecuadamente
app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navbar,
        dbc.Container(id="page-content", className="mb-4", fluid=True),
    ],
    style={
        'backgroundColor': '#4f5382',  # Fondo de color
        'backgroundImage': f'url(data:image/png;base64,{imagen_codificada})',
        'backgroundSize': 'cover',  # Ajusta la imagen para que toda se vea sin recortes
        'backgroundPosition': 'center',
        'backgroundRepeat': 'no-repeat',  # Evita la repetición de la imagen
        'height': '100vh',  # Ocupa el 100% de la altura de la ventana
        'width': '100vw',   # Ocupa el 100% del ancho de la ventana
        'overflow': 'auto',  # Evita el scroll si hay contenido adicional
        'position': 'fixed',  # Fija la imagen de fondo
    }
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/":
        return home_layout
    elif pathname == "/data_upload":
        return data_upload_layout
    elif pathname == "/book_now":
        return layout_book
    elif pathname == "/eda":
        return layout_eda  
    else:
        return dbc.Card(
            dbc.CardBody(
                [
                    html.H1("404: Not found", className="text-danger"),
                    html.Hr(),
                    html.P(f"The pathname {pathname} was not recognized..."),
                ]
            ),
            className="mt-3"
        )

@app.callback(
    Output("navbar-collapse", "is_open"),
    [Input("navbar-toggler", "n_clicks")],
    [State("navbar-collapse", "is_open")]
)
def toggle_navbar_collapse(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=True, host='0.0.0.0', port=9000) # <- To Dockerize the Dash
