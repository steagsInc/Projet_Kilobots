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

os.chdir("/home/mohamed/PycharmProjects/Projet_Kilobots")
print("Chemin avant lancement du serveur : ", os.getcwd())

S = swarmDescriptor("morphogenesis")


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
"""D = dict(
                data=dict(
                    id = i,
                    label = "",
                    U = df["U"][i],
                    V = df["V"][i],
                    nb_neighbours = 0
                ),
                position = dict(
                    x=df["x"][i],
                    y=df["y"][i]
                )
            ) for i in range(0,df.shape[0])"""

position_robots = cyto.Cytoscape(
        id='robots-positions',
        layout={'name': 'preset'},
        style={'width': '100%', 'height': '500px'},
        elements=[],
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
            'selector': '[U>'+str(S.seuillage_turing_spots)+']',
            'style': {
                'background-color': 'GREEN'
            }
        },

    ]
)




app.layout = html.Div([
    html.H1("Visualisation des simulations"),
    position_robots,
    html.Button("Executer Simulation",id="Lancer",n_clicks=0)
    ]
)

@app.callback(Output('robots-positions',"elements"),[Input("Lancer","n_clicks")])
def executer_simulation(n):
    S.executeSimulation()
    print("click")
    return [
        dict(
            data=dict(
                id=i,
                label="",
                U=S.concentrations[i][0],
                V=S.concentrations[i][1],
                nb_neighbours=0
            ),
            position=dict(
                x=S.positions[i],
                y=S.positions[i]
            )
        ) for i in range(0, S.positions.shape[0])
    ]
    pass


if __name__ == '__main__':

    print("Chemin avant lancement du serveur : ",os.getcwd())
    app.run_server(debug=True)


