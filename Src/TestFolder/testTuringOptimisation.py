import os
import numpy as np
import cma

from Src.simulationController.topologyOptimizer import topologyOptimisation

print("Début du test de l'extracteur des propriétés de l'essaim sur le chemin : ", os.getcwd())
os.chdir("../..")
S = topologyOptimisation("pile",nb=300,visible=True,time=2000)


def fitnessShapeIndex(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    S.Swarm.shapeIndex()
    print("Pénalité de ",-S.Swarm.SumshapeIndex())
    return -S.Swarm.SumshapeIndex()

def fitnessRectanglitude(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    S.Swarm.shapeIndex()
    print("Pénalité de ",-S.Swarm.SumshapeIndex())
    return -np.array(S.Swarm.rectanglitude()).sum()

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

    return -np.std(S.Swarm.concentrations,axis=0).sum()+np.mean(np.max(S.Swarm.concentrations,0))



def fitnessTuringSpot(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    #S.Swarm.shapeIndex()
    print("Pénalité de ",-S.Swarm.nb_turing_spots)
    S.Swarm.calculerTuringSpots(seuil=4)
    #print("Précision : ",S.getPrecision())
    return -S.Swarm.nb_turing_spots

def computeOptization(func,iter):
    S = topologyOptimisation("pile", nb=250, visible=False, time=500,model=[])
    w = S.extract_genotype()
    S.Swarm.controller.withVisiblite(True)
    res = cma.CMAEvolutionStrategy(w, 1).optimize(func, maxfun=iter).result
    S.put_genotype(res[0])
    S.Swarm.controller.write_params(S.Swarm.controller.read_params())



if(__name__=="__main__"):
    w = S.extract_genotype()
    S.Swarm.controller.rez_params()
    S.Swarm.controller.withVisiblite(True)
    S.computeSimulation()
    S.Swarm.calculerTuringSpots(seuil=4)
    print("Depart a : ", S.Swarm.nb_turing_spots)
    res = cma.CMAEvolutionStrategy(w, 0.0001).optimize(fitnessTuringSpot, maxfun=500).result
    S.put_genotype(res[0])
    S.Swarm.controller.write_params(S.Swarm.controller.read_params())



#NTM MILO J'AI VRAIMENT UN COMMIT POUR CA
#ICI MILO, VOUS ME RECEVEZ ?