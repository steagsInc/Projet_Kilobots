import dash_daq as daq
import dash_cytoscape as cyto
import dash
import dash_html_components as html
import json
import os
import numpy as np
import pandas as pd
import dash_core_components as dcc
from dash.dependencies import Input, Output

from Test_Sensibility.Visualisation_Voisinage import external_stylesheets

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    [
        cyto.Cytoscape(
            id='robots-positions',
            layout={'name': 'preset'},
            style={'width': '100%', 'height': '500px'},
            elements=[

            ],
            stylesheet=[
                {
                    'selector': 'node',
                    'style': {
                        'background-color': 'BLACK',
                        'label': 'data(U)'
                    }
                },
                {
                    'selector': '[U<0]',
                    'style': {
                        'background-color': 'RED'
                    }
                },
                {
                    'selector': '[U>200]',
                    'style': {
                        'background-color': 'GREEN'
                    }
                },

            ]

        )


    ]


)



