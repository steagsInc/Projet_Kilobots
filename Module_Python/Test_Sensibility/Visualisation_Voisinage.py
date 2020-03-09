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
from dash.dependencies import Input, Output
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.tree import DecisionTreeClassifier

from Test_Sensibility.Fonctions import readStates, setTopology, simulatePerceptron

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
if(os.getcwd().split("/")[-1] == "Test_Sensibility"):
    os.chdir("..")

batch_size = 50

number_robots = 10
topology_robots = "random"
courant_id = None
simulationData = None
robots_for_data = 0
X_Df = None


simulationDone = False
modeles_calcules = {}
simulatePerceptron(topology_robots,number_robots)

print("Début read")

df = pd.DataFrame(readStates())
print("Fin read")
df = df.rename(columns={0: "x", 1: "y",2: "U", 3: "V"})

X_data = np.zeros(df.shape)

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

models_dropdown = dcc.Dropdown(
        id="model-selection",
        options=[
            {'label': 'Multilayer Perceptron', 'value': 'MLP'},
            {'label': 'SVM', 'value': 'SVM'},
            {'label': 'Decision Tree', 'value': 'DT'},
            {'label': 'Linear Stochastic Gradient Descent', 'value': 'SGDC'},
            {'label': 'Random Forest', 'value': 'RF'},
            {'label': 'AdaBoost', 'value': 'AdaBoost'},
            {'label': 'Gradient Adaboost', 'value': 'GradientBoostClassification'},
        ],
        multi=True,
        searchable=True,
    )

positions_robots = cyto.Cytoscape(
        id='robots-positions',
        layout={'name': 'preset'},
        style={'width': '100%', 'height': '500px'},
        elements=[
            dict(
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
            ) for i in range(0,df.shape[0])
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

plot2_data = [
                {
                    'x': [1, 2, 3, 4],
                    'y': [4, 1, 3, 5],
                    'text': ['a', 'b', 'c', 'd'],
                    'customdata': ['c.a', 'c.b', 'c.c', 'c.d'],
                    'name': 'Trace 1',
                    'mode': 'markers',
                    'marker': {'size': 12}
                },
                {
                    'x': [1, 2, 3, 4],
                    'y': [9, 4, 1, 4],
                    'text': ['w', 'x', 'y', 'z'],
                    'customdata': ['c.w', 'c.x', 'c.y', 'c.z'],
                    'name': 'Trace 2',
                    'mode': 'markers',
                    'marker': {'size': 12}
                }
            ]


plot2 = dcc.Graph(
        id='visualisation-classes',
        figure={
            'data':plot2_data,
            'layout': {
                'clickmode': 'event+select'
            }
        }
    )

L = html.Div(
    [
    dcc.Graph(
        id='robots',
        figure={
            'data': [
                dict(
                    x=df["U"],
                    y=df['V'],
                    text="Random Text",
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 20,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name="P"
                )
            ],
            'layout': dict(
                xaxis={'type': 'log', 'title': 'GDP Per Capita'},
                yaxis={'title': 'Life Expectancy'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    ),
        html.Label('Slider'),
        dcc.Slider(
            min=0,
            max=9,
            marks={i : 'Label {}'.format(i) if i == 1 else str(i) for i in range(1, 9)},
            value=5,
        )
    ]

)

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


T1 = html.Div([
    positions_robots
    ,
    dcc.Dropdown(
        id="robots-topology",
        options=[
        {'label': 'Ligne droite', 'value': 'line'},
        {'label': 'Elipse', 'value': 'ellipse'},
        {'label': 'Cercle', 'value': 'pile'},
        {'label': 'Aléatoire', 'value': 'random'}
        ],
    value='pile'
    ),
    dcc.Slider(
        id="robots-number",
        min=0,
        max=500,
        marks={i: 'Label {}'.format(i) if (i%75==0) else "" for i in range(500)},
        value=15,
    ),
    dcc.Slider(
        id="robots-range",
        min=0,
        max=300,
        marks={i: 'Label {}'.format(i) if (i % 20 == 0) else "" for i in range(500)},
        value=55,
    ),
    html.Pre(id='cytoscape-tapNodeData-json'),
    html.Div(
        [
        dcc.Input(
            id="Nombre de simulations",
            placeholder='Enter a value...',
            type='text',
            value=''
        ),


        ]
    )
])


T2 = html.Div([
    html.H3("Nombre de voisins pris en compte pour la prédiction : "),
    daq.NumericInput(
    id='number-neighbours',
    value=5,
    max=10,
    min=0
    ),
    html.Button('Générer les données', id='train-modele'),
    html.Button('Display Data', id='display-modele'),
    html.Button('Plot 2D', id='plot2-modele'),
    html.Button('Entrainer des modèles et évaluer leur précision', id='training'),
    html.H3("Choix du modèle a tester"),
    models_dropdown,
    html.Div(id ='Table'),
    html.H3("Progression des simulations"),
    html.Div(
        [
            html.H4("Fin des simulations "),
            daq.Indicator(
                id='simulation-indication',
                value=False,
                color="red"
            ),
        ]
    ),
    html.Div([plot2]),
    daq.LEDDisplay(
      id='modele-precision',
      value="3.14159"
    ),
    dcc.Graph(
        id="models-comparing",
        figure=dict(
            data=[],
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
    )


])

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Tab one', value='tab-1',children=T1),
        dcc.Tab(label='Tab two', value='tab-2',children=T2),
    ])
])



#@app.callback([Output("simulation-indication","color"),Output("Table","children")],[Input("number-neighbours","value")])
#def resetIndicator(v):
#    global robots_for_data
#    robots_for_data = v
#    return "red",pd.DataFrame(np.array([]))


@app.callback(Output("simulation-indication","color"),
              [Input("robots-positions","elements"),Input('train-modele',"n_clicks"),Input("number-neighbours","value")])
def generateTable(elem,nclicks,val):
    global X_data
    global X_Df
    global robots_for_data
    global simulationDone
    global modeles_calcules
    sizes = np.random.randint(number_robots,2*number_robots,batch_size)
    p = 0
    X_data = []
    data = elem
    neighbours = []
    cols = []
    if(robots_for_data != val):
        robots_for_data = val
        simulationDone = False
        modeles_calcules = {}
        return "red"
    for i in data:
        if(i["data"]["id"]==courant_id):
            neighbours = i["data"]["voisins"]
    if (len(neighbours) != 0):
        neighbours = np.array(neighbours)
        neighbours = np.random.choice(neighbours,robots_for_data)
        neighbours = list(neighbours)

    for s in sizes:
        p = p + 1
        if(p%2 == 0):
            topology = topology_robots
        else:
            topology = "random"
        simulatePerceptron(topology,s)
        X = readStates()
        X = np.array(X)
        X = X[:,-2:]
        if(len(neighbours) != 0):
            d = X[np.array([courant_id]+neighbours),:]
            cumul = []
            cols = []
            for i in d:
                cumul = cumul + list(i)
            cumul.append(1 if topology == topology_robots else 0)
            for i in range(0,d.shape[0]):
                cols = cols + [k+str(i) for k in ["U Voisin ","V Voisin"]]
            cols = cols + [topology_robots]
            X_data = X_data + [cumul]
    X_data = np.array(X_data)
    Dat = pd.DataFrame(X_data)
    D = { i : cols[i] for i in range(0,len(cols))}
    X_Df = Dat.rename(columns = D)
    simulationDone = True
    return "green"


@app.callback(Output("Table","children"),[Input('display-modele',"n_clicks")])
def displayTable(n):
    return generate_table(pd.DataFrame(X_Df))


@app.callback(Output("visualisation-classes","figure"),[Input('plot2-modele',"n_clicks")])
def displayPlot(n):
    X,Y = X_data[:,:-1], X_data[:,-1]
    id_1 = np.where(Y.reshape(-1,1) == 1)[0]
    print("Test de id1 : ",id_1)
    id_0 = np.where(Y.reshape(-1,1)== 0)[0]
    print("Test de id1 : ",id_0)
    P = PCA(n_components=2)
    X = P.fit_transform(X)
    X_1 = X[id_1]
    X_0 = X[id_0]
    print("X_0 : ",pd.DataFrame(X_0))
    plot2_data = [
        {
            'x': list(X_0[:,0]),
            'y': list(X_0[:,1]),
            'name': "Random",
            'mode': 'markers',
            'marker': {'size': 12}
        },
        {
            'x': list(X_1[:,0]),
            'y': list(X_1[:,1]),
            'name': topology_robots,
            'mode': 'markers',
            'marker': {'size': 10}
        }
    ]
    return {
            'data':plot2_data,
            'layout': {
                'clickmode': 'event+select'
            }
        }




@app.callback(Output('cytoscape-tapNodeData-json', 'children'),
              [Input('robots-positions', 'tapNodeData')])
def displayTapNodeData(data):
    return json.dumps(data, indent=2)



@app.callback(
    [
        Output("robots-positions","elements"),
        Output("number-neighbours","max"),
    ],
    [
        Input('robots-positions', 'tapNodeData'),
        Input("robots-number","value"),
        Input("robots-topology","value"),
        Input("robots-range","value")
    ]
)
def draw_edges(selected,number,topology,rnge):
    L= []
    global simulationData
    global topology_robots
    global number_robots
    global courant_id
    R = []
    topology_robots = topology
    number_robots = number
    simulationData = readStates()

    simulatePerceptron(topology, number)
    df = pd.DataFrame(readStates())
    df = df.rename(columns={0: "x", 1: "y", 2: "U", 3: "V"})
    n = np.array(df)
    n = n[:, 0:2]
    for i in range(0,df.shape[0]):
        ni = n[i]
        dist = n - ni
        dist = np.sqrt(np.power(dist[:, 0], 2) + np.power(dist[:, 1], 2))
        id = np.where(dist <= rnge)[0]
        id = id[np.where(id != i)[0]]
        R = R + [dict(
                data=dict(
                    id=i,
                    label="",
                    U=df["U"][i],
                    V=df["V"][i],
                    nb_neighbours = id.shape[0],
                    voisins = list(id)
                ),
                position=dict(
                    x=df["x"][i],
                    y=df["y"][i]
                )
            )]
        if(selected!=None and i == int(selected["id"])):
            courant_id = int(selected["id"])
            L = [dict(data=dict(source=str(i), target=str(j))) for j in id]
            R = R + L
    return R,len(L)



@app.callback([Output("modele-precision","value"),Output("models-comparing","figure")],[Input('training','n_clicks'),Input('model-selection','value')])
def train_print(n,modeles):
    global modeles_calcules
    figure = dict(
        data=[],
        layout=dict(
            title='Comparaison des modèles : ' + ", ".join(modeles_calcules.keys()),
            showlegend=True,
            legend=dict(
                x=0,
                y=1.0
            ),
            margin=dict(l=40, r=0, t=40, b=30)
        )
    )
    if(modeles == None):
        return np.NAN,figure
    X, Y = X_data[:, :-1], X_data[:, -1]
    P = MLPClassifier()
    print("Etat des modèles : ",modeles)
    for i in modeles:
        if(i in modeles_calcules):
            continue
        if(i == "MLP"):
            P = MLPClassifier()
        if(i == "SVM"):
            P = LinearSVC()
        if(i == "DT"):
            P = DecisionTreeClassifier()
        if(i == "SGDC"):
            P = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
        if(i == "RF"):
            P = RandomForestClassifier(n_estimators=10)
        if( i== "AdaBoost"):
            P =  AdaBoostClassifier(n_estimators=100)
        if(i == "GradientBoostClassification"):
            P = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        P.fit(X,Y)
        rez = learning_curve(P, X, Y,train_sizes=np.linspace(0.1, 1.0, 10), cv = 5)
        print("Rez = ",rez)
        x = rez[0]
        train = np.mean(rez[1],axis=1)
        test = np.mean(rez[2],axis=1)
        modeles_calcules[i]= P, x , test

    data = [
        dict(
            x=modeles_calcules[i][1],
            y=modeles_calcules[i][2],
            name=i,
            marker=dict(
                color=np.random.rand(3,)
            )
        ) for i in modeles_calcules
    ]
    figure = dict(
        data=data,
        layout=dict(
            title='Comparaison des modèles : '+", ".join(modeles_calcules.keys()),
            showlegend=True,
            legend=dict(
                x=0,
                y=1.0
            ),
            margin=dict(l=40, r=0, t=40, b=30)
        )
    )
    return str(np.array(cross_val_score(P,X,Y)).mean()),figure




if __name__ == '__main__':
    print("Chemin avant lancement du serveur : ",os.getcwd())
    app.run_server(debug=True)



