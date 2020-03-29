# -*- coding: utf-8 -*-
import json
import os
import numpy as np
import pandas as pd
import dash_daq as daq
import dash_cytoscape as cyto
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier

from Src.controllers.swarmDescriptor import swarmDescriptor

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#os.chdir("/home/mohamed/PycharmProjects/Projet_Kilobots")
if (os.getcwd().split("/")[-1] == "Dashboard"):
    os.chdir("../..")
print("Chemin avant lancement du serveur : ", os.getcwd())

S = swarmDescriptor("morphogenesis")
S.setTime(250)
S.controller.withVisiblite(False)
S.setTopology("pile")
S.seuillage_turing_spots = 2

variables_model = ('A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v')

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

graphe = dcc.Graph(
    figure=dict(
        data=[
            dict(
                x=[],
                y=[],
                name='Rest of world',
                marker=dict(
                    color='rgb(55, 83, 109)'
                )
            ),
        ],
        layout=dict(
            title='US Export of Plastic Scrap',
            showlegend=True,
            legend=dict(
                x=0,
                y=1.0
            ),
            margin=dict(l=40, r=0, t=40, b=30)
        )
    ),
    style={'height': 300},
    id='sensibility-graph'
)



app.layout = html.Div([
    dcc.Dropdown(
        id = "topology_depart",
        options=[
            dict(
                label = m,
                value = m
            ) for m in variables_model
        ],
        multi=True,
        value="A_VAL"
    ),
    dcc.Slider(
        id="sigma",
        min=0,
        max=3,
        marks={i: 'sigma={}'.format(str(i)[0:4]) for i in np.linspace(-3,3,30)},
        value=0.01,
        step=0.01,
    ),
    dcc.Slider(
            id="mean",
            min=-1,
            max=2,
            marks={i: 'sigma={}'.format(str(i)[0:4]) for i in np.linspace(0,3,20)},
            value=0.01,
            step=0.01,

    ),
]
)




if __name__ == '__main__':
    #print("Chemin avant lancement du serveur : ",os.getcwd())
    app.run_server(debug=True)


