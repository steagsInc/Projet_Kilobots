import os
import numpy as np
import cma

from Src.simulationController.topologyOptimizer import topologyOptimisation

print("Début du test de l'extracteur des propriétés de l'essaim sur le chemin : ", os.getcwd())
os.chdir("../..")
S = topologyOptimisation("line",nb=50,visible=False,time=200)


def fitnessShapeIndex(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    S.Swarm.shapeIndex()
    print("Pénalité de ",-S.Swarm.SumshapeIndex())
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

def varianceUV(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.readDatas()
    print("Variance : ",np.std(S.Swarm.concentrations,axis=0))
    print("Moyenne : ",S.Swarm.concentrations.mean(axis=0))

    print("Donc la pénalité ",-np.std(S.Swarm.concentrations,axis=0).sum())
    return -np.std(S.Swarm.concentrations,axis=0).sum()+ S.Swarm.concentrations.mean(axis=0).T@S.Swarm.concentrations.mean(axis=0)



def fitness(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    #S.Swarm.shapeIndex()
    print("Pénalité de ",-S.Swarm.nb_turing_spots)
    S.Swarm.calculerTuringSpots(seuil=4)
    #print("Précision : ",S.getPrecision())
    return -S.Swarm.nb_turing_spots


if(__name__=="__main__"):
    w = S.extract_genotype()
    S.Swarm.controller.withVisiblite(True)
    res = cma.CMAEvolutionStrategy(w, 1).optimize(varianceUV, maxfun=2000).result
    S.put_genotype(res[0])
    S.Swarm.controller.write_params(S.Swarm.controller.read_params())
