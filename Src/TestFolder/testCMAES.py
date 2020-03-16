import os

import cma
import tensorflow as tf
import matplotlib.pyplot as plt

import time
from Src.simulationController.predictionAccuracyMeasurement import simulator

print("Début du test de l'extracteur des propriétés de l'essaim sur le chemin : ", os.getcwd())
os.chdir("../..")
S = simulator(nb=30)

S.Swarm.setTime(100)

meilleur_precision = 0
meilleur_fitness = 1000
nb_exec = 0
historique_precisions = []
x_precisions = []
historique_fitness = []

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
    L = S.getHinge()
    historique_precisions.append(P)
    historique_fitness.append(L)
    if(P > meilleur_precision):
        meilleur_precision = P
        print("Précision : ", P)
    if (L < meilleur_fitness):
        meilleur_fitness = L
        print("Penalite : ", L)
    print("tps ecoulé : ",time.time()-timeStart)
    return S.getLogLoss()

if(__name__=="__main__"):
    S.Swarm.controller.put_Random_Weights()
    w = S.Swarm.controller.read_Weights()
    with tf.device("GPU:0"):
        res = cma.CMAEvolutionStrategy(w, 0.1).optimize(fitness, maxfun=100).result
    plt.plot(x_precisions, historique_precisions, color='green')
    plt.plot(x_precisions, historique_fitness, color='red')
    plt.show()
    print("Fin de l'executin : meilleur précision : ",meilleur_precision)
