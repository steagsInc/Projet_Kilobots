import os

import cma
import numpy as np
import pandas as pd
import json
from sklearn.metrics import log_loss,hinge_loss

os.chdir('..')
path_weights = "kilombo/templates/kilotron/weights.txt"
path_kilotron = "kilombo/templates/kilotron"
nb_bot_max = 50
concordance = ["circle","line"]


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
    os.system("./kilotron 2> erreur.tmp")
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
        y_pred = np.zeros((len(d["bot_states"]),2))
        for i in range(len(d["bot_states"])):
            y_pred[i][0] = d["bot_states"][i]["state"]["p1"]
            y_pred[i][1] = d["bot_states"][i]["state"]["p2"]
    os.chdir("../../..")
    return y_pred


historique_y = []



def testAccuracy(w,borne = 10):
    tests = []
    writeWeights(w)
    L = []
    cpt = 0
    for i in range(0,borne):
        if(i%2 == 0):
            topology = "circle"
        else:
            topology = "line"
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

#Voila la version pour deux prédictions
def testAccuracyFixed(w,borne = 2):
    tests = []
    writeWeights(w)
    L = []
    cpt = 0
    for i in range(0,borne):
        if(i%2 == 0):
            topology = "circle"
        else:
            topology = "line"
        setTopology(topology,nb_bot_max)
        simulatePerceptron(topology,nb_bot_max)
        y_p = getPredictions()
        print(y_p)
        input("Interface bloqué pour que tu visualises les prédiction appuie sur entrée batard")
        #En gros la je construit une ligne de zeros a cahque iteration
        Z = np.zeros(y_p.shape[1])
        #Ensuite j'inscris dessus la bonne prédiction
        Z[i%2] = 1
        c = np.zeros((1,len(y_p[0])))
        for j in range(y_p.shape[0]):
            #Je compte une pénalité si l'avis du robot donc argmax(p) != de la vraie topologie
            if(np.argmax(y_p[j]) != i%2):
                c= c + (y_p[j] - Z)**2
        c = np.array(c)
        cpt = cpt + c.sum()
        print('Sanction pour une topologie : ',c.sum())
    print("Pénalité totale : ",cpt)
    return cpt

#La différence avec celui d'avance c'est que la je compte juste les robots qui se sont trompés
#de prédiction qu'ils se soient trompé de 1 ou de 1 milliard c'est pareil
def testAccuracyFixedV2(w,borne = 2):
    tests = []
    writeWeights(w)
    L = []
    cpt = 0
    for i in range(0,borne):
        if(i%2 == 0):
            topology = "circle"
        else:
            topology = "line"
        setTopology(topology,nb_bot_max)
        simulatePerceptron(topology,nb_bot_max)
        y_p = getPredictions()
        #En gros la je construit une ligne de zeros a cahque iteration
        Z = np.zeros(y_p.shape[1])
        #Ensuite j'inscris dessus la bonne prédiction
        Z[i%2] = 1
        c = 0
        for j in range(y_p.shape[0]):
            #Je compte une pénalité si l'avis du robot donc argmax(p) != de la vraie topologie
            if(np.argmax(y_p[j]) != i%2):
                c= c + 1
        cpt = cpt + c
        print('Sanction pour une topologie : ',c)
    print("Pénalité totale : ",cpt)
    return cpt


def testLogLoss(w,borne = 2):
    tests = []
    writeWeights(w)
    L = []
    cpt = 0
    for i in range(0,borne):
        if(i%2 == 0):
            topology = "circle"
        else:
            topology = "line"
        setTopology(topology,nb_bot_max)
        simulatePerceptron(topology,nb_bot_max)
        y_p = getPredictions()
        Z = np.hstack((np.zeros((y_p.shape[0], 1)), np.ones((y_p.shape[0], 1))))
        if(i%2 == 1):
            Z = np.hstack((np.ones((y_p.shape[0],1)),np.zeros((y_p.shape[0],1)) ))

        for k in range(y_p.shape[0]):
            y_p[k,:] = y_p[k,:] / np.sum(y_p[k,:])
        print(y_p)
        c=log_loss(Z,y_p)
        cpt = cpt + c
        print('Sanction pour une topologie : ',c)
    print("Pénalité totale : ",cpt)
    return cpt

def testHingeLoss(w,borne = 2):
    tests = []
    writeWeights(w)
    L = []
    cpt = 0
    for i in range(0,borne):
        if(i%2 == 0):
            topology = "circle"
        else:
            topology = "line"
        setTopology(topology,nb_bot_max)
        simulatePerceptron(topology,nb_bot_max)
        y_p = getPredictions()
        Z = -np.ones(y_p.shape[0]).astype(int)
        if(i%2 == 1):
            Z = np.ones(y_p.shape[0]).astype(int)
        P = np.max(y_p,axis=1)
        P[np.where(np.argmax(y_p,axis=1) == 0)] = -P[np.where(np.argmax(y_p,axis=1) == 0)]
        c=hinge_loss(P,Z)
        cpt = cpt + c
        print('Sanction pour une topologie : ',c)
    print("Pénalité totale : ",cpt)
    return cpt


def testCMMLoss(w,borne = 2):
    tests = []
    writeWeights(w)
    L = []
    cpt = 0
    for i in range(0,borne):
        if(i%2 == 0):
            topology = "circle"
        else:
            topology = "line"
        setTopology(topology,nb_bot_max)
        simulatePerceptron(topology,nb_bot_max)

        y_p = getPredictions()
        Z = -np.ones(y_p.shape[0]).astype(int)
        if(i%2 == 1):
            Z = np.ones(y_p.shape[0]).astype(int)
        P = np.max(y_p,axis=1)
        P[np.where(np.argmax(y_p,axis=1) == 0)] = -P[np.where(np.argmax(y_p,axis=1) == 0)]
        c=hinge_loss(P,Z)
        cpt = cpt + c
        print('Sanction pour une topologie : ',c)
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
        Y = np.argmax(y_p)
        print("VECTEUR PREDICTIONS : ",Y)
        input("-----")
        T = [1 if Y[j] == concordance.index(topology) else 0 for j in range(len(Y))]
        T = np.array(T)
        p = p + np.sum(T) / T.shape[0]
    print("Précision globale de ", p / 10)


def putRandomWeights():
    w = readWeights()
    r = np.random.random(w.shape[0])
    writeWeights(r)



if (__name__=="__main__"):
    print("chemin courant : ",os.getcwd())
    putRandomWeights()
    w = readWeights()
    res = cma.CMAEvolutionStrategy(w, 0.0001).optimize(testAccuracyFixed, maxfun=20).result
    #best_w = res[0]
