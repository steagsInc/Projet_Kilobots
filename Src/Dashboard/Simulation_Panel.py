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
changed = False

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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
    html.Center(
        [
            html.Button("Executer Simulation",id="Lancer",n_clicks=0),
            daq.Indicator(
                id='fin_simulation',
                value=False,
                color="green"
            )
        ]
    ),
    html.Button("Rendu Visuel",id="render",n_clicks=0),
    html.H3("Temps de simulation : "),
    dcc.Slider(
                    id="gestion_temps",
                    min=0,
                    max=2000,
                    marks={i: '{} s'.format(i) for i in range(0,2000,100)},
                    value=5,
                ),
    html.H3("Taille de l'essaim : "),
        dcc.Slider(
                        id="taille_essaim",
                        min=0,
                        max=500,
                        marks={i: '{} robots'.format(i) for i in range(0,500,25)},
                        value=5,
                    ),
    html.H3("Topology de départ : "),
    dcc.Dropdown(
        id = "topology_depart",
        options=[
            {'label': 'Cercle', 'value': 'circle'},
            {'label': 'Ligne', 'value': 'line'},
            {'label': 'Aléatoire', 'value': 'random'},
            {'label': 'Empilé', 'value': 'pile'},
            {'label': 'Elipse', 'value': 'ellipse'},
        ],
        multi=False,
        value="pile"
    ),
    html.P(id="placeholder_time"),
    html.P(id="placeholder_changement"),
    html.P(id="placeholder_topology"),
    html.P(id="placeholder_taille"),
    html.P(id="placeholder_render")

]
)

@app.callback(Output("placeholder_time","children"),[Input("gestion_temps","value")])
def maj_temps(valeur):
    global changed
    S.setTime(valeur)
    changed = True
    return []

@app.callback(Output("placeholder_topology","children"),[Input("topology_depart","value")])
def maj_topology(valeur):
    global changed
    S.setTopology(valeur)
    changed = True
    return []



@app.callback(Output("placeholder_taille","children"),[Input("taille_essaim","value")])
def maj_taille(valeur):
    global changed
    S.setNb_robots(valeur)
    changed = True
    return []


@app.callback(Output("fin_simulation","color"),[Input("placeholder_time","children"),Input('robots-positions',"elements"),Input("placeholder_topology","children"),Input("placeholder_taille","children")])
def changeColor(a1,a2,a3,a4):
    if changed:
        return "red"
    else:
        return "green"


@app.callback(Output('placeholder_render',"children"),[Input("render","n_clicks")])
def render(n):
    S.controller.withVisiblite(True)
    S.executeSimulation()
    S.controller.withVisiblite(False)



@app.callback(Output('robots-positions',"elements"),[Input("Lancer","n_clicks")])
def executer_simulation(n):
    global changed
    S.executeSimulation()
    changed = False
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
                x=S.positions[i][0],
                y=S.positions[i][1]
            )
        ) for i in range(0, S.positions.shape[0])
    ]


if __name__ == '__main__':
    print("Chemin avant lancement du serveur : ",os.getcwd())
    app.run_server(debug=True)


