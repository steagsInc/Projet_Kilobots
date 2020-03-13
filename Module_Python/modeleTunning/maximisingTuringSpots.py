import cma

from interface_simulation import lire_params, write_params, execute_simulation
import os
import numpy as np

from topologyEvaluation.metricsTSCPSI import countTuringSpots, shapeIndex

os.chdir("..")


def extract_genotype(model=None):
    if model is None:
        model = ['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v']
    P = lire_params()
    w = []
    for i in model:
        w.append(P[i])
    w  = np.array(w)
    return w


def put_genotype(w,model=None):

    if model is None:
        model = ['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v']
    P = lire_params()
    for i in range(len(model)):
        P[model[i]] = w[i]
    write_params(P)



def turingSpotObjectif(w,model=None):
    if model is None:
        model =['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v']
    #TODO : Remove this
    print("chemin courant : ",os.getcwd())
    os.chdir("/home/mohamed/PycharmProjects/Projet_Kilobots/Module_Python")
    put_genotype(w,model)
    execute_simulation()
    return -countTuringSpots()

def shapeIndexObjectif(w,model=None):
    if model is None:
        model =['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v']
    #TODO : Remove this
    print("chemin courant : ",os.getcwd())
    os.chdir("/home/mohamed/PycharmProjects/Projet_Kilobots/Module_Python")
    put_genotype(w,model)
    execute_simulation()
    return -shapeIndex()


def productObjectif(w,model=None):
    if model is None:
        model =['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v']
    #TODO : Remove this
    print("chemin courant : ",os.getcwd())
    put_genotype(w,model)
    execute_simulation()
    return -(shapeIndex()*10*countTuringSpots())

def poderatedSum(w,model=None):
    if model is None:
        model =['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v']
    #TODO : Remove this
    put_genotype(w,model)
    execute_simulation()
    if(countTuringSpots() == 0):
        return 1000*shapeIndex()+-35*shapeIndex()-2*countTuringSpots()+sumUV("produced/endstate.json")**2
    return -35*shapeIndex()-2*countTuringSpots()+sumUV("produced/endstate.json")**2

def objectif1(w,model=None):
    if model is None:
        model =['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v']
    #TODO : Remove this
    put_genotype(w,model)
    execute_simulation()
    if(countTuringSpots() == 0):
        return 1000*shapeIndex()
    return -35*shapeIndex()-2*countTuringSpots()





def sumUV(jsonfile):
    import json
    cpt = 0
    with open(jsonfile,"r") as fp:
        data = json.load(fp)
        for i in data["bot_states"]:
            print(i["state"])
            for j in i["state"]:
                cpt = cpt +  i["state"][j]*i["state"][j]

    return float(cpt)



if (__name__=="__main__"):
    print("Chemin au debut : ",os.getcwd())
    w = extract_genotype(model =['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v'])
    res = cma.CMAEvolutionStrategy(w, 1).optimize(poderatedSum, maxfun=100).result
    best_w = res[0]
    for i in res:
        print(i)
    put_genotype(best_w, model =['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v'])
    execute_simulation()


