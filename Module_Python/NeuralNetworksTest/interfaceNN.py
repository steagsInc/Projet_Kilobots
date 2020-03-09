import os

import cma
import numpy as np
import pandas as pd
import json


os.chdir('..')
path_weights = "kilombo/templates/kilotron/weights.txt"
path_kilotron = "kilombo/templates/kilotron"

concordance = ["circle","pile"]


def setTopology(topology,number=0):
    os.chdir(path_kilotron)
    with open("kilombo.json","r") as fp:
        d = json.load(fp)
        fp.close()
    d["formation"] = topology
    if(number):
        d["nBots"] = str(number)
    with open("kilombo.json", "w") as fp:
        json.dump(d, fp)
    os.chdir("../../..")

def simulatePerceptron(topology,number):
    os.chdir(path_kilotron)
    os.system("make")
    os.chdir("../../..")

    setTopology(topology,number)

    os.chdir(path_kilotron)
    os.system("./kilotron.c")
    os.chdir("../../..")


def readWeights():
    with open(path_weights,"r") as fp:
        c = fp.readlines()
        w = np.zeros(len(c))
        for i in range(len(c)):
            w[i] = float(c[i].strip())
    return w

def writeWeights(w):
    with open(path_weights,"w") as fp:
        st = ""
        for i in w:
            st = st + str(i) + "\n"
        fp.write(st)





def getPredictions():
    os.chdir(path_kilotron)
    with open("endstate.json","r") as fp:
        d = json.load(fp)
        y_pred = np.zeros(len(d["bot_states"]))
        for i in range(len(d["bot_states"])):
            y_pred[i] = d["bot_states"][i]["state"]["p"]
    os.chdir("../../..")
    return y_pred


historique_y = []



def testAccuracy(w,borne = 10,nb_bot_max = 10):
    tests = []
    writeWeights(w)
    L = []
    cpt = 0
    for i in range(0,borne):
        if(i%2 == 0):
            topology = "circle"
        else:
            topology = "pile"
        setTopology(topology,nb_bot_max)
        simulatePerceptron(topology,nb_bot_max)
        y_p = getPredictions()
        c = [0  for i in range(0,y_p.shape[0])]
        for i in range(y_p.shape[0]):
            c[i] = c[i] + (y_p[i] - concordance.index(topology))**2
        c = np.array(c)
        cpt = cpt + c.sum()
        print('Sanction : ',c.sum())
        tests.append(c.sum() / c.shape[0])
        L.append(c.sum() / c.shape[0])
    historique_y.append(L)
    print("Pénalité totale : ",cpt)
    return cpt


def testAccuracy(w,borne = 10,nb_bot_max = 10):
    tests = []
    writeWeights(w)
    L = []
    cpt = 0
    for i in range(0,borne):
        if(i%2 == 0):
            topology = "circle"
        else:
            topology = "pile"
        setTopology(topology,nb_bot_max)
        simulatePerceptron(topology,nb_bot_max)
        y_p = getPredictions()
        c = [0  for i in range(0,y_p.shape[0])]
        for i in range(y_p.shape[0]):
            c[i] = c[i] + (y_p[i] - concordance.index(topology))**2
        c = np.array(c)
        cpt = cpt + c.sum()
        print('Sanction : ',c.sum())
        tests.append(c.sum() / c.shape[0])
        L.append(c.sum() / c.shape[0])
    historique_y.append(L)
    print("Pénalité totale : ",cpt)
    return cpt


def validation(nb_bot_max):
    p = 0
    for i in range(0,20):
        if(i%2 == 0):
            topology = "circle"
        else:
            topology = "pile"
        setTopology(topology,nb_bot_max)
        simulatePerceptron(topology,nb_bot_max)
        y_p = getPredictions()
        Y = [1 if y_p[j] > 0.5 else 0 for j in range(y_p.shape[0])]
        print("VECTEUR PREDICTIONS : ",Y)
        input("-----")
        T = [1 if Y[j] == concordance.index(topology) else 0 for j in range(len(Y))]
        T = np.array(T)
        p = p + np.sum(T) / T.shape[0]
    print("Précision globale de ", p / 10)




if (__name__=="__main__"):
    print("chemin courant : ",os.getcwd())
    w = readWeights()
    res = cma.CMAEvolutionStrategy(w, 1).optimize(testAccuracy, maxfun=20).result
    best_w = res[0]
    validation(20)
    #print("Historique de prédictions ",historique_y)
