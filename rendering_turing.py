"""
Ces methodes sont utilises afin de ne pas surcharger les notebooks.
"""
import os

import cma
import numpy as np
from Src.simulationController.topologyOptimizer import topologyOptimisation
import matplotlib.pyplot as plt

best_fitness = np.inf


print("Debut du test de l'extracteur des proprietes de l'essaim sur le chemin : ", os.getcwd())
S = topologyOptimisation("pile",nb=300,visible=False,time=2500)
#S.computeSimulation()


def renduFitness(nom_fitness,nb_iterations,sigma=0.2,folder=None):
    S.Swarm.setTime(1500)
    S.Swarm.controller.rez_params()
    if(nom_fitness == "Turing Spot"):
        computeOptization(fitnessTuringSpot,nb_iterations,sigma)
    elif(nom_fitness == "Shape Index"):
        computeOptization(fitnessShapeIndex,nb_iterations,sigma)
    elif(nom_fitness == "Aggregation Multiplication"):
        computeOptization(fitnessAggregation,nb_iterations)
    elif (nom_fitness == "Rectanglitude"):
        computeOptization(fitnessRectanglitude, nb_iterations)
    S.Swarm.readDatas()
    S.Swarm.calculerTuringSpots(seuil=4)
    S.Swarm.shapeIndex()
    f = plt.figure()
    f.clear()
    plt.close(f)
    S.Swarm.renduTuringSpot()
    if(folder):
        S.Swarm.genererRenduTSFolder(folder)


def renduModel(nom_fitness,nb_iterations,sigma=0.2,Model=None,folder=None):
    S.Swarm.setTime(1500)
    S.Swarm.controller.rez_params()
    if(Model):
        S.model = Model
    if(nom_fitness == "Turing Spot"):
        computeOptization(fitnessTuringSpot,nb_iterations,sigma)
    elif(nom_fitness == "Shape Index"):
        computeOptization(fitnessShapeIndex,nb_iterations,sigma)
    elif(nom_fitness == "Aggregation Multiplication"):
        computeOptization(fitnessAggregation1,nb_iterations)
    elif (nom_fitness == "Aggregation Addition"):
        computeOptization(fitnessAggregation2, nb_iterations)
    elif (nom_fitness == "Rectanglitude"):
        computeOptization(fitnessRectanglitudeMaxSTS, nb_iterations)
    S.Swarm.readDatas()
    S.Swarm.calculerTuringSpots(seuil=4)
    S.Swarm.shapeIndex()
    f = plt.figure()
    f.clear()
    plt.close(f)

    if (folder):
        S.Swarm.genererRenduTSFolder(folder)
    S.Swarm.renduTuringSpot()



def fitnessShapeIndex(w):
    global best_fitness
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.readDatas()
    S.Swarm.shapeIndex()
    if (-np.array(S.Swarm.shapeIndex()).sum() < best_fitness):
        print("Ameliorations du shape index : ",S.Swarm.shapeIndex())
        plt.clf()
        S.Swarm.genererRendu()
        best_fitness = -np.array(S.Swarm.shapeIndex()).sum()
    return -np.array(S.Swarm.shapeIndex()).sum()

def fitnessRectanglitude(w):
    global best_fitness
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.readDatas()
    S.Swarm.rectanglitude()
    if (-np.array(S.Swarm.rectanglitude()).sum() < best_fitness):
        print("Ameliorations du shape index : ",S.Swarm.rectanglitude())
        plt.clf()
        S.Swarm.genererRendu()
        best_fitness = -np.array(S.Swarm.rectanglitude()).sum()
    return -np.array(S.Swarm.rectanglitude()).sum()


def fitnessRectanglitudeMean(w):
    global best_fitness
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.readDatas()
    S.Swarm.rectanglitude()
    if (-np.array(S.Swarm.rectanglitude()).sum() < best_fitness):
        print("Ameliorations du shape index : ",S.Swarm.rectanglitude())
        plt.clf()
        S.Swarm.genererRendu()
        best_fitness = -np.array(S.Swarm.rectanglitude()).sum()
    return -np.array(S.Swarm.rectanglitude()).mean()


def fitnessRectanglitudeMeanTS(w):
    global best_fitness
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.readDatas()
    S.Swarm.rectanglitude()
    S.Swarm.calculerTuringSpots(seuil=4)
    if (-np.array(S.Swarm.rectanglitude()).sum() < best_fitness):
        print("Ameliorations du shape index : ",S.Swarm.rectanglitude())
        plt.clf()
        S.Swarm.genererRendu()
        best_fitness = -np.array(S.Swarm.rectanglitude()).sum()
    return -(np.array(S.Swarm.rectanglitude()).mean()*S.Swarm.nb_turing_spots)

def fitnessRectanglitudeMaxSTS(w):
    global best_fitness
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.readDatas()
    S.Swarm.rectanglitude()-(2*np.array(S.Swarm.rectanglitude()).mean()*S.Swarm.nb_turing_spots+20*S.Swarm.nb_turing_spots)
    S.Swarm.calculerTuringSpots(seuil=4)
    if (-(2*np.array(S.Swarm.rectanglitude()).mean()*S.Swarm.nb_turing_spots+20*S.Swarm.nb_turing_spots)< best_fitness):
        print("Rectanglitude max : ",np.array(S.Swarm.rectanglitude()).max())
        print("nb turing spots : ",S.Swarm.nb_turing_spots)
        plt.clf()
        S.Swarm.genererRendu()
        best_fitness = -(2*np.array(S.Swarm.rectanglitude()).mean()*S.Swarm.nb_turing_spots+20*S.Swarm.nb_turing_spots)
    return -(2*np.array(S.Swarm.rectanglitude()).mean()*S.Swarm.nb_turing_spots+20*S.Swarm.nb_turing_spots)

def fitnessAggregation(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    S.Swarm.shapeIndex()
    plt.clf()
    #print("Nombre de turing de ",-S.Swarm.nb_turing_spots)
    S.Swarm.calculerTuringSpots(seuil=4)
    #print("Precision : ",S.getPrecision())
    #print("Shape de ", -S.Swarm.SumshapeIndex())
    print("Penalite de  : ",-(S.Swarm.nb_turing_spots*S.Swarm.SumshapeIndex()))
    return -(S.Swarm.nb_turing_spots*S.Swarm.SumshapeIndex())

def varianceUV(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.readDatas()
    print("Variance : ",np.std(S.Swarm.concentrations,axis=0))
    return -np.std(S.Swarm.concentrations,axis=0).sum()+np.mean(np.max(S.Swarm.concentrations,0))


def fitnessAggregation1(w):
    global best_fitness
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.readDatas()
    S.Swarm.shapeIndex()
    S.Swarm.calculerTuringSpots(seuil=4)
    if(-(S.Swarm.nb_turing_spots*np.array(S.Swarm.shapeIndex()).sum()) < best_fitness and S.Swarm.nb_turing_spots > 0):
        S.Swarm.renduTuringSpot()
        best_fitness = -(S.Swarm.nb_turing_spots*np.array(S.Swarm.shapeIndex()).sum())
        print("Nouvelle meilleur fitness a : ",best_fitness)
    return -(S.Swarm.nb_turing_spots*np.array(S.Swarm.shapeIndex()).sum())


def fitnessAggregation2(w):
    global best_fitness
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.readDatas()
    S.Swarm.shapeIndex()
    S.Swarm.calculerTuringSpots(seuil=4)
    if(-(S.Swarm.nb_turing_spots+np.array(S.Swarm.shapeIndex()).sum()) < best_fitness and S.Swarm.nb_turing_spots > 0):
        S.Swarm.renduTuringSpot()
        best_fitness = -(S.Swarm.nb_turing_spots+np.array(S.Swarm.shapeIndex()).sum())
        print("Nouvelle meilleur fitness a : ",best_fitness)
    return -(S.Swarm.nb_turing_spots+np.array(S.Swarm.shapeIndex()).sum())

def fitnessTuringSpot(w):
    global best_fitness
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.readDatas()
    S.Swarm.calculerTuringSpots(seuil=4)
    if(-S.Swarm.nb_turing_spots < best_fitness and S.Swarm.nb_turing_spots > 0):
        print("Amelioration du nombre de turing Spot : ", S.Swarm.nb_turing_spots)
        S.Swarm.renduTuringSpot()
        best_fitness = -S.Swarm.nb_turing_spots
    return -S.Swarm.nb_turing_spots

def computeOptization(func,iter,sigma = 0.1):
    S.Swarm.controller.rez_params()
    w = S.extract_genotype()
    S.Swarm.controller.withVisiblite(True)
    res = cma.CMAEvolutionStrategy(w, sigma).optimize(func, maxfun=iter).result
    S.put_genotype(res[0])
    S.Swarm.controller.write_params(S.Swarm.controller.read_params())


if(__name__=="__main__"):
    S.Swarm.controller.rez_params()
    renduModel("Turing Spot",550,sigma=0.01)
    #renduFitness("Shape Index",50)