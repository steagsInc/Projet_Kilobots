# -*- coding: utf-8 -*-
# ! /usr/bin/env python3

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

# os.chdir("/home/mohamed/PycharmProjects/Projet_Kilobots")
if (os.getcwd().split("/")[-1] == "Dashboard"):
    os.chdir("../..")
print("Chemin avant lancement du serveur : ", os.getcwd())

S = swarmDescriptor("morphogenesis")
S.setTime(250)
S.controller.withVisiblite(False)
S.setTopology("pile")
S.seuillage_turing_spots = 2
changed = False
polar_th = 4
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
params = S.controller.parametres_model
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
            'selector': '[U>' + str(polar_th - 3) + ']',
            'style': {
                'background-color': 'CYAN'
            }
        },
        {
            'selector': '[U>' + str(polar_th - 2) + ']',
            'style': {
                'background-color': 'BLUE'
            }
        },
        {
            'selector': '[U>' + str(polar_th - 1) + ']',
            'style': {
                'background-color': 'PINK'
            }
        },
        {
            'selector': '[U>=' + str(polar_th) + ']',
            'style': {
                'background-color': 'GREEN'
            }
        }

    ]
)

app.layout = html.Div([
    html.H1("Visualisation des simulations"),
    position_robots,
    html.Center(
        [
            html.Button("Executer Simulation", id="Lancer", n_clicks=0),
            daq.Indicator(
                id='fin_simulation',
                value=False,
                color="green"
            )
        ]
    ),
    html.Button("Rendu Visuel", id="render", n_clicks=0),
    html.H3("Temps de simulation : "),
    dcc.Slider(
        id="gestion_temps",
        min=0,
        max=15000,
        marks={i: '{} s'.format(i) for i in range(0, 15000, 1000)},
        value=5,
    ),
    html.H3("Taille de l'essaim : "),
    dcc.Slider(
        id="taille_essaim",
        min=0,
        max=500,
        marks={i: '{} robots'.format(i) for i in range(0, 500, 25)},
        value=5,
    ),
    html.H3("Topology de départ : "),
    dcc.Dropdown(
        id="topology_depart",
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
    html.P(id="placeholder_render"),
    html.P(id="placeholder_U"),
    html.P(id="placeholder_V"),
    html.P(id="placeholder_A"),
    html.P(id="placeholder_B"),
    html.P(id="placeholder_C"),
    html.P(id="placeholder_D"),
    html.P(id="placeholder_E"),
    html.P(id="placeholder_F"),
    html.Button("Remettre les paramètres d'origine", id="rez_params", n_clicks=0),
    html.H4("A : "),
    dcc.Input(id="input3", type="number", placeholder="", debounce=True),
    html.H4("B : "),
    dcc.Input(id="input4", type="number", placeholder="", debounce=True),
    html.H4("C : "),
    dcc.Input(id="input5", type="number", placeholder="", debounce=True),
    html.H4("D : "),
    dcc.Input(id="input6", type="number", placeholder="", debounce=True),
    html.H4("E : "),
    dcc.Input(id="input7", type="number", placeholder=""),
    html.H4("F : "),
    dcc.Input(id="input8", type="number", placeholder="", debounce=True),
    html.H4("U : "),
    dcc.Input(id="input1", type="number", placeholder=""),
    html.H4("V : "),
    dcc.Input(id="input2", type="number", placeholder="", debounce=True),

]
)

A_VAL = 0.08
B_VAL = -0.08
C_VAL = 0.03
D_VAL = 0.03
E_VAL = 0.1
F_VAL = 0.12
G_VAL = 0.06
D_u = 0.5
D_v = 10


@app.callback(
    [
        Output("input1", "value"),
        Output("input2", "value"),
        Output("input3", "value"),
        Output("input4", "value"),
        Output("input5", "value"),
        Output("input6", "value"),
        Output("input7", "value"),
        Output("input8", "value"),
    ],
    [Input("rez_params", "n_clicks")])
def rez(valeur):
    return D_u, D_v, A_VAL, B_VAL, C_VAL, D_VAL, E_VAL, F_VAL


@app.callback(Output("placeholder_U", "children"), [Input("input1", "value")])
def majU(valeur):
    if (valeur):
        S.controller.write_params({"D_u": valeur})


@app.callback(Output("placeholder_V", "children"), [Input("input2", "value")])
def majV(valeur):
    if (valeur):
        params.update({"D_v": valeur})
        S.controller.write_params(params)


@app.callback(Output("placeholder_A", "children"), [Input("input3", "value")])
def majA(valeur):
    if (valeur):
        params.update({"A_VAL": valeur})
        S.controller.write_params(params)


@app.callback(Output("placeholder_B", "children"), [Input("input4", "value")])
def majB(valeur):
    if (valeur):
        params.update({"B_VAL": valeur})
        S.controller.write_params(params)


@app.callback(Output("placeholder_C", "children"), [Input("input5", "value")])
def majC(valeur):
    if (valeur):
        params.update({"C_VAL": valeur})
        S.controller.write_params(params)


@app.callback(Output("placeholder_D", "children"), [Input("input6", "value")])
def majD(valeur):
    if (valeur):
        params.update({"D_VAL": valeur})
        S.controller.write_params(params)


@app.callback(Output("placeholder_E", "children"), [Input("input7", "value")])
def majE(valeur):
    if (valeur):
        params.update({"E_VAL": valeur})
        S.controller.write_params(params)


@app.callback(Output("placeholder_F", "children"), [Input("input8", "value")])
def majF(valeur):
    if (valeur):
        params.update({"F_VAL": valeur})
        S.controller.write_params(params)


@app.callback(Output("placeholder_time", "children"), [Input("gestion_temps", "value")])
def maj_temps(valeur):
    global changed
    S.setTime(valeur)
    changed = True
    return []


@app.callback(Output("placeholder_topology", "children"), [Input("topology_depart", "value")])
def maj_topology(valeur):
    global changed
    S.setTopology(valeur)
    changed = True
    return []


@app.callback(Output("placeholder_taille", "children"), [Input("taille_essaim", "value")])
def maj_taille(valeur):
    global changed
    S.setNb_robots(valeur)
    changed = True
    return []


@app.callback(Output("fin_simulation", "color"),
              [Input("placeholder_time", "children"), Input('robots-positions', "elements"),
               Input("placeholder_topology", "children"), Input("placeholder_taille", "children")])
def changeColor(a1, a2, a3, a4):
    if changed:
        return "red"
    else:
        return "green"


@app.callback(Output('placeholder_render', "children"), [Input("render", "n_clicks")])
def render(n):
    S.controller.withVisiblite(True)
    S.executeSimulation()
    S.controller.withVisiblite(False)


@app.callback(Output('robots-positions', "elements"), [Input("Lancer", "n_clicks")])
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
    # pca = PCA(n_components=2, svd_solver='full'
    print("Chemin avant lancement du serveur : ", os.getcwd())
    app.run_server(debug=True)
