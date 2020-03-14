import os

import cma

from Src.simulationController.topologyOptimizer import topologyOptimisation

print("Début du test de l'extracteur des propriétés de l'essaim sur le chemin : ", os.getcwd())
os.chdir("../..")
S = topologyOptimisation("pile",nb=150,visible=False)


def fitnessShapeIndex(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    S.Swarm.shapeIndex()
    print("Pénalité de ",-S.Swarm.SumshapeIndex())
    #print("Précision : ",S.getPrecision())
    return -S.Swarm.SumshapeIndex()



def fitnessAggregation(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    S.Swarm.shapeIndex()
    #print("Nombre de turing de ",-S.Swarm.nb_turing_spots)
    S.Swarm.calculerTuringSpots(seuil=4)
    #print("Précision : ",S.getPrecision())
    #print("Shape de ", -S.Swarm.SumshapeIndex())
    print("Penalité de  : ",-(S.Swarm.nb_turing_spots*S.Swarm.SumshapeIndex()))
    return -(S.Swarm.nb_turing_spots*S.Swarm.SumshapeIndex())

def fitness(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    S.Swarm.shapeIndex()
    print("Pénalité de ",-S.Swarm.nb_turing_spots)
    S.Swarm.calculerTuringSpots()
    #print("Précision : ",S.getPrecision())
    return -S.Swarm.nb_turing_spots


if(__name__=="__main__"):
    w = S.extract_genotype()
    res = cma.CMAEvolutionStrategy(w, 0.2).optimize(fitnessAggregation, maxfun=200).result
    S.put_genotype(res[0])
    S.Swarm.controller.write_params(S.Swarm.controller.read_params())
