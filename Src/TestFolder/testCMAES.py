import os

import cma

from Src.predictionAccuracyMeasurement import simulator

print("Début du test de l'extracteur des propriétés de l'essaim sur le chemin : ", os.getcwd())
os.chdir("../..")
S = simulator()


def fitness(w):
    S.Swarm.controller.write_Weights(w)
    S.computeSimulation()
    print("Pénalité de ",S.getHinge())
    print("Précision : ",S.getPrecision())
    return S.getLeastSquare()

if(__name__=="__main__"):
    S.Swarm.controller.put_Random_Weights()
    w = S.Swarm.controller.read_Weights()
    res = cma.CMAEvolutionStrategy(w, 0.01).optimize(fitness, maxfun=200).result
