import os

import cma
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from Src.simulationController.predictionAccuracyMeasurement import simulator

print("Debut du test de l'extracteur des proprietes de l'essaim sur le chemin : ", os.getcwd())
S = simulator(nb=50)

S.Swarm.controller = S.Swarm.controller.withVisiblite(True).withTime(100).withNombre(10)
meilleur_precision = 0
meilleur_fitness = 1000
nb_exec = 0
historique_precisions = []
x_precisions = []
historique_fitness = []
best_w = None
def fitness(w):
    timeStart = time.time()
    global meilleur_precision
    global meilleur_fitness
    global historique_precisions
    global historique_fitness
    global nb_exec
    global x_precisions
    nb_exec = nb_exec + 1
    print("Etape ", nb_exec , " : ")
    x_precisions.append(nb_exec)
    S.Swarm.controller.write_Weights(w)
    S.computeSimulation()
    P = S.getPrecision()
    L = S.maxHinge()
    historique_precisions.append(P)
    historique_fitness.append(L)
    if(P > meilleur_precision):
        meilleur_precision = P
        print("Precision : ", P)
        best_w = w
    if (L < meilleur_fitness):
        meilleur_fitness = L
        print("Penalite : ", L)
    #print("tps ecoule : ",time.time()-timeStart)
    return L


def fitnessHinge(w):
    timeStart = time.time()
    global meilleur_precision
    global meilleur_fitness
    global historique_precisions
    global historique_fitness
    global nb_exec
    global x_precisions
    nb_exec = nb_exec + 1
    print("Etape ", nb_exec , " : ")
    x_precisions.append(nb_exec)
    S.Swarm.controller.write_Weights(w)
    S.computeSimulation()
    P = S.getPrecision()
    L = S.getHinge()
    historique_precisions.append(P)
    historique_fitness.append(L)
    if(P > meilleur_precision):
        meilleur_precision = P
        print("Precision : ", P)
        best_w = w
    if (L < meilleur_fitness):
        meilleur_fitness = L
        print("Penalite : ", L)
    #print("tps ecoule : ",time.time()-timeStart)
    return L

def fitnessLeastSquare(w):
    timeStart = time.time()
    global meilleur_precision
    global meilleur_fitness
    global historique_precisions
    global historique_fitness
    global nb_exec
    global x_precisions
    nb_exec = nb_exec + 1
    print("Etape ", nb_exec , " : ")
    x_precisions.append(nb_exec)
    S.Swarm.controller.write_Weights(w)
    S.computeSimulation()
    P = S.getPrecision()
    L = S.getLeastSquare()
    historique_precisions.append(P)
    historique_fitness.append(L)
    if(P > meilleur_precision):
        meilleur_precision = P
        print("Precision : ", P)
        best_w = w
    if (L < meilleur_fitness):
        meilleur_fitness = L
        print("Penalite : ", L)
    #print("tps ecoule : ",time.time()-timeStart)
    return L

def fitnessLogLoss(w):
    timeStart = time.time()
    global meilleur_precision
    global meilleur_fitness
    global historique_precisions
    global historique_fitness
    global nb_exec
    global x_precisions
    nb_exec = nb_exec + 1
    print("Etape ", nb_exec , " : ")
    x_precisions.append(nb_exec)
    S.Swarm.controller.write_Weights(w)
    S.computeSimulation()
    P = S.getPrecision()
    L = S.getLogLoss()
    historique_precisions.append(P)
    historique_fitness.append(L)
    if(P > meilleur_precision):
        meilleur_precision = P
        print("Precision : ", P)
        best_w = w
    if (L < meilleur_fitness):
        meilleur_fitness = L
        print("Nouvelle meilleure loss : ", L)
    #print("tps ecoule : ",time.time()-timeStart)
    return L


def fitnessPrecision(w):
    timeStart = time.time()
    global meilleur_precision
    global meilleur_fitness
    global historique_precisions
    global historique_fitness
    global nb_exec
    global x_precisions
    nb_exec = nb_exec + 1
    print("Iteration ", nb_exec, " : ")
    S.Swarm.controller.write_Weights(w)
    S.computeSimulation()
    P = S.getPrecision()
    x_precisions.append(nb_exec)
    historique_fitness.append(P)
    if (P > meilleur_precision):
        meilleur_precision = P
        print("Nouvelle meilleure Precision : ", P)
    # print("tps ecoule : ",time.time()-timeStart)
    return -P

def fitnessCrossEntropy(w):
    timeStart = time.time()
    global meilleur_precision
    global meilleur_fitness
    global historique_precisions
    global historique_fitness
    global nb_exec
    global x_precisions
    nb_exec = nb_exec + 1
    print("Iteration ", nb_exec, " : ")
    S.Swarm.controller.write_Weights(w)
    S.computeSimulation()
    L = S.getCrossEntropy()
    P = S.getPrecision()
    x_precisions.append(nb_exec)
    historique_fitness.append(P)
    if(L < meilleur_fitness):
        meilleur_fitness = L
        print("Loss : ", L)
    if (P > meilleur_precision):
        meilleur_precision = P
        print("Nouvelle meilleure Precision : ", P)
    # print("tps ecoule : ",time.time()-timeStart)
    return L



def optimizeNeuralNetwork(iter,sigma,func,shape):
    global nb_exec
    global x_precisions
    global historique_fitness
    global historique_precisions
    S.Swarm.controller.setShape(shape)
    historique_fitness.clear()
    historique_precisions.clear()
    x_precisions.clear()
    nb_exec = 0
    S.Swarm.controller.put_Random_Weights()
    w = S.Swarm.controller.read_Weights()
    res = cma.CMAEvolutionStrategy(w, sigma).optimize(func, maxfun=iter).result
    plt.plot(x_precisions, historique_fitness, color='red')
    plt.show()



def analysePredictions(shape,iter=5,init_param = (0,1),verbose=False):
    S.Swarm.controller.setShape(shape)
    temps = []
    SUP = np.zeros((2,2))
    SPP = 0
    SSTD = np.zeros((1,2))
    for i in range(0,iter):
        S.Swarm.controller.put_Random_Weights(mean=init_param[0],sigma=init_param[1])
        t = time.time()
        S.computeSimulation()
        t = time.time() - t
        temps.append(t)
        p = list(S.Swarm.predictions.T)+list(S.Swarm.concentrations.T)
        p = np.array(p)
        C = np.cov(p)
        UP = np.array([[C[0,2],C[0,3]],[C[1,2],C[1,3]]])
        PP = np.array(C[0,1])
        SPP = SPP + PP
        SUP = SUP + UP
        SSTD = SSTD + np.std(S.Swarm.predictions,axis=0)
    temps = np.array(temps)
    SPP = SPP / temps.shape[0]
    SUP = SUP / temps.shape[0]
    SSTD = SSTD / temps.shape[0]
    temps = np.mean(temps)
    if(verbose):
        print("Resultats moyens fait sur ",iter," itterations.")
        print("MATRICE DE COVARIANCE P1,P2 -> U,V :")
        print(SUP)
        print("COVARIANCE P1-> P2 :",SPP)
        print("VARIANCE P1 , P2 : ",SSTD)
        print("TEMPS MOYEN : ",temps)
    return SSTD,temps

def plot_variance_neuronnes(min,max,hidden = 2,points = 20):
    I = np.linspace(min,max,points)
    Y = []
    T = []
    for s in I:
        shape = dict(
            HIDDEN = hidden,
            NEURONES = int(s)
        )
        t  = analysePredictions(shape, 1)
        Y.append(t[0][0])
        T.append(t[1])
        print("Simulation pour une shape de : ",s, " finie en : ",t[1])

    plt.plot(I,Y)
    plt.show()
    plt.plot(I,T)
    plt.show()

if(__name__=="__main__"):
    shape = dict(
        NEURONES=100,
        HIDDEN=2
    )
    #plot_variance_neuronnes(10,150,3,50)
    optimizeNeuralNetwork(1, 0.8, fitnessPrecision, shape)
    #optimizeNeuralNetwork(10, 0.001, fitnessCrossEntropy, shape)
