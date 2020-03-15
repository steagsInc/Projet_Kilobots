import os

import cma
import tensorflow as tf

from Src.simulationController.predictionAccuracyMeasurement import simulator

print("Début du test de l'extracteur des propriétés de l'essaim sur le chemin : ", os.getcwd())
os.chdir("../..")
S = simulator(nb=10)

def fitness(w):
    S.Swarm.controller.write_Weights(w)
    S.computeSimulation()
    #print("Pénalité de ",S.getHinge())
    print("Précision : ",S.getPrecision())
    return S.getHinge()

if(__name__=="__main__"):
    S.Swarm.controller.put_Random_Weights()
    w = S.Swarm.controller.read_Weights()
    with tf.device('/GPU:0'):
        res = cma.CMAEvolutionStrategy(w, 0.1).optimize(fitness, maxfun=500).result
