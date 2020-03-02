import cma
from cma import purecma

from interface_simulation import lire_params, write_params, execute_simulation
import os
import numpy as np

from topologyEvaluation.metricsTSCPSI import countTuringSpots, shapeIndex


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
        model = ['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v']
    #TODO : Remove this
    print("chemin courant : ",os.getcwd())
    os.chdir("/home/mohamed/PycharmProjects/Projet_Kilobots/Module_Python")
    put_genotype(w,model)
    execute_simulation()
    return -shapeIndex()


def productObjectif(w,model=None):
    if model is None:
        model = ['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v']
    #TODO : Remove this
    print("chemin courant : ",os.getcwd())
    os.chdir("/home/mohamed/PycharmProjects/Projet_Kilobots/Module_Python")
    put_genotype(w,model)
    execute_simulation()
    return -(shapeIndex()*10*countTuringSpots())

def poderatedSum(w,model=None):
    if model is None:
        model=['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL']
    #TODO : Remove this
    print("chemin courant : ",os.getcwd())
    os.chdir("/home/mohamed/PycharmProjects/Projet_Kilobots/Module_Python")
    put_genotype(w,model)
    execute_simulation()
    if(countTuringSpots() == 0):
        return shapeIndex()
    return -(10*shapeIndex()+countTuringSpots())



if (__name__=="__main__"):
    os.chdir("/home/mohamed/PycharmProjects/Projet_Kilobots/Module_Python")
    w = extract_genotype(model=['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL'])
    res = cma.CMAEvolutionStrategy(w, 1).optimize(poderatedSum, maxfun=50).result
    best_w = res[0]
    for i in res:
        print(i)

    put_genotype(best_w, model=['A_VAL', 'B_VAL', 'C_VAL'])
    execute_simulation()


