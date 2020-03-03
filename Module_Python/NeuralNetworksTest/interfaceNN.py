import os
import numpy as np
import json
os.chdir('..')
path_weights = "kilombo/templates/kilotron/weights.txt"
path_kilotron = "kilombo/templates/kilotron"

concordance = ["pile","line"]


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


def setTopology(topology):
    print(os.getcwd())
    with open("kilombo.json","r") as fp:
        d = json.load(fp)
        fp.close()
    d["formation"] = topology
    with open("kilombo.json", "w") as fp:
        json.dump(d, fp)

def getPredictions():
    os.chdir(path_kilotron)
    with open("endstate.json","r") as fp:
        d = json.load(fp)
        y_pred = np.zeros(len(d["bot_states"]))
        for i in range(len(d["bot_states"])):
            y_pred[i] = d["bot_states"][i]["state"]["p"]
            print(d["bot_states"][i]["state"]["p"])
    os.chdir("../../..")
    return y_pred




def simulatePerceptron(topology,number):
    os.chdir(path_kilotron)
    os.system("make")
    setTopology(topology)
    os.system("./kilotron")
    os.chdir("../../..")

#simulatePerceptron("pile")
#simulatePerceptron("line")
getPredictions()