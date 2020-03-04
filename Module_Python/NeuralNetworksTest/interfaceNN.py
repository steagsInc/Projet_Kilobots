import os

import cma
import numpy as np
import json
os.chdir('..')
path_weights = "kilombo/templates/kilotron/weights.txt"
path_kilotron = "kilombo/templates/kilotron"

concordance = ["circle","pile"]


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


def setTopology(topology,number=0):
    os.chdir(path_kilotron)
    with open("kilombo.json","r") as fp:
        d = json.load(fp)
        fp.close()
    d["formation"] = topology
    if(number):
        d["nBots"] = number
    with open("kilombo.json", "w") as fp:
        json.dump(d, fp)
    os.chdir("../../..")



def getPredictions():
    os.chdir(path_kilotron)
    with open("endstate.json","r") as fp:
        d = json.load(fp)
        y_pred = np.zeros(len(d["bot_states"]))
        for i in range(len(d["bot_states"])):
            y_pred[i] = 1 if d["bot_states"][i]["state"]["p"]>0.5 else 0
    os.chdir("../../..")
    return y_pred




def simulatePerceptron(topology,number):
    os.chdir(path_kilotron)
    os.system("make")
    os.chdir("../../..")

    setTopology(topology)

    os.chdir(path_kilotron)
    os.system("./kilotron")
    os.chdir("../../..")

def testAccuracy(w,borne = 10,nb_bot_max = 10):
    tests = []
    writeWeights(w)
    for i in range(0,borne):
        if(i%2 == 0):
            topology = "circle"
        else:
            topology = "pile"
        setTopology(topology,nb_bot_max)
        simulatePerceptron(topology,nb_bot_max)
        y_p = getPredictions()
        y_p[ np.where(y_p > 1)] = 1
        c = [1  if (y_p[i] == concordance.index(topology) )else 0 for i in range(0,y_p.shape[0])]
        c = np.array(c)
        tests.append(c.sum() / c.shape[0])
    tests = np.array(tests)
    print("Moyenne en tests : ",tests.mean())
    return -tests.mean() + np.sum(w)


if (__name__=="__main__"):
    w = readWeights()
    res = cma.CMAEvolutionStrategy(w, 1).optimize(testAccuracy, maxfun=10).result
    best_w = res[0]
    print("Meilleur résultat : ",best_w)
