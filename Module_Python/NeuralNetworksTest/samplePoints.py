import pandas as pd
import numpy as np
import os
import json



topologies = ["line","circle"]
path = "kilombo/templates/kilotron"
#path_kilotron = "kilombo/templates/kilotron"

path_kilotron = "produced"
path = "produced"

distance = lambda x1,y1,x2,y2 : np.sqrt((x1-x2)**2+(y1-y2)**2)

def simulatePerceptron1(topology,number):
    os.chdir(path_kilotron)
    os.system("make")
    os.chdir("../../..")

    setTopology(topology,number)

    os.chdir(path_kilotron)
    os.system("./kilotron")
    os.chdir("../../..")


def simulatePerceptron(topology,number):
    os.chdir(path_kilotron)
    os.system("make")
    os.chdir("../../..")

    setTopology(topology,number)

    os.chdir(path_kilotron)
    os.system("./morphogenesis.c")
    os.chdir("../../..")

def generate_neighbours(id,X,neighborhood_max = 55):
    neighbours = []
    for i in range(X.shape[0]):
        if(i != id):
            print(distance(X[i][0],X[i][1],X[id][0],X[id][1]))
            if(distance(X[i][0],X[i][1],X[id][0],X[id][1]) <= neighborhood_max ):
                neighbours.append(i)
    neighbours = np.array(neighbours)
    if(neighbours.shape[0] != 0 ):
        return X[neighbours]
    else:
        return None


def generateSamplesOfSimulations(id, n_sizes, max_size = 250,neighborhood_max = 10):
    sizes = np.random.randint(10,max_size,n_sizes)
    X =  readStates()
    Total = []
    for s in sizes:
        for j in range(len(topologies)):
            simulatePerceptron(topologies[j],s)
            N = generate_neighbours(id,X,neighborhood_max)
            print("N = ",N)
            if(N.any() != None):
                N = np.hstack( (N,j+np.ones((N.shape[0],1))) )
                Total = Total + list(N)
    return np.array(Total)




def setTopology(topology,number=0):
    os.chdir(path)
    with open("kilombo.json","r") as fp:
        d = json.load(fp)
        fp.close()
    d["formation"] = topology
    if(number):
        d["nBots"] = str(number)
    with open("kilombo.json", "w") as fp:
        json.dump(d, fp)
    os.chdir("../../..")


def readStates():
    os.chdir("..")
    print(os.getcwd())

    os.chdir(path)
    with open("endstate.json", "r") as fp:
        d = json.load(fp)["bot_states"]
        X = []
        for bot in d:
            L = []
            L.append(bot["x_position"])
            L.append(bot["y_position"])
            for s in bot['state']:
                L.append(bot['state'][s])
            X.append(L)
        os.chdir("../../..")
    X = np.array(X)
    return X

X = generateSamplesOfSimulations(0,15,neighborhood_max=150)
print(X)
X,Y = X[:,:-1],X[:,-1]
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
L = MLPClassifier()
L.fit(X,Y)
print(cross_val_score(L,X,Y))