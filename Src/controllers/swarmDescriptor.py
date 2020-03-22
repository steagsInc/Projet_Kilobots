import math
import os
import json

from Src.milolib import topologyCalculs
from Src.milolib.topologyCalculs import countTuringSpotsWithGraph, multiClusterShapeIndex, clustersRectanglitude
from Src.controllers.simulationController import simulationController
import matplotlib.pyplot as plt
import numpy as np



class Cluster:
    def __init__(self,ids):
        self.cluster_id = None
        self.nb_element = 0
        self.ids = ids
        self.shape_index = 0
        self.rectanglitude = 0

    def __str__(self):
        return str(
            dict(
                id = self.cluster_id,
                size=self.nb_element,
                shape_index=self.shape_index,
                rectangliude = self.rectanglitude,
                elements = self.ids
            )
        )


class swarmDescriptor:

    def __init__(self,path):
        #Données qui conditionnent la simulation
        self.path = path
        self.controller = simulationController(path)
        topologyCalculs.fp = os.getcwd() + "/embedded_simulateurs/" + self.path + "/endstate.json"
        #Données récupérées de la simulation
        self.data = {}
        self.states = []
        #Données traitées a partir des sorties de la simulation
        self.predictions = None
        self.positions = None
        self.concentrations = None
        #TODO Implémenter ça
        self.canaux = None
        self.portee_communication  = 40
        #Données construites par les calculs
        self.clusters = []
        self.nb_turing_spots = 0
        self.turing_spots = []
        self.seuillage_turing_spots = 4

    def setRange(self,portee):
        self.portee_communication = portee


    def setTopology(self,topology):
        self.topology = topology
        self.controller = self.controller.withTopology(topology)


    def readDatas(self,verbose=False):
        with open("embedded_simulateurs/" + self.path + "/" + "endstate.json", "r") as file1:
            self.data = json.load(file1)["bot_states"]
            predictions = []
            positions = []
            concentrations = []
            for i in self.data:
                self.states.append(i["state"])
                positions.append(np.array([i["x_position"],i["y_position"]]))
                concentrations.append(np.array([i["state"]["u"],i["state"]["v"]]))
                if("p1" in i["state"].keys()):
                    predictions.append(np.array([i["state"]["p1"],i["state"]["p2"]]))
            self.positions = np.array(positions)
            self.predictions = np.array(predictions)
            self.concentrations = np.array(concentrations)
            if(verbose):
                print("Positions : ",self.positions)
                print("Concentations : ",self.concentrations)
                print("Predictions : ", self.predictions)


    def clusteriser(self):
        self.clusters.clear()
        L = countTuringSpotsWithGraph(distVoisin=self.controller.range, valueTresh=0)
        cpt = 0
        for i in L:
            cpt = cpt + 1
            C = Cluster(i)
            C.nb_element = len(i)
            C.cluster_id = cpt
            self.clusters.append(C)

    def clustersizes(self):
        if(len(self.clusters) == 0):
            self.clusteriser()
        return np.array([c.nb_element for c in self.clusters])

    def getDonnesCluster(self,predOrStates=False):
        D = {}
        self.clusteriser()
        for c in self.clusters:
            if(predOrStates):
                if(self.predictions.shape[0]>0):
                    D[c.cluster_id] = self.predictions[c.ids]
            else:
                indexes = np.array(c.ids)
                D[c.cluster_id] = self.concentrations[indexes]
        return D

    def getSommeUV(self):
        return np.sum(self.concentrations,axis=0)


    def calculerTuringSpots(self,seuil = 5):
        topologyCalculs.fp = os.getcwd() + "/embedded_simulateurs/" + self.path + "/endstate.json"
        self.seuillage_turing_spots = seuil
        self.turing_spots = countTuringSpotsWithGraph(distVoisin=self.controller.range,valueTresh=self.seuillage_turing_spots)
        self.nb_turing_spots = len(self.turing_spots)
        return self.turing_spots

    def shapeIndex(self):
        self.clusteriser()
        topologyCalculs.fp = os.getcwd() + "/embedded_simulateurs/" + self.path + "/endstate.json"
        cpt = 0
        C = multiClusterShapeIndex(show=False, distVoisin=self.controller.range)
        for i in C:
            if (len(self.clusters) > cpt):
                self.clusters[cpt].shape_index = i
                cpt = cpt + 1
        return multiClusterShapeIndex(show=False, distVoisin=self.controller.range)


    def rectanglitude(self, seuil = 0):
        self.clusteriser()
        topologyCalculs.fp = os.getcwd() + "/embedded_simulateurs/" + self.path + "/endstate.json"


        """
        cpt = 0
        C = clustersRectanglitude( distVoisin=self.controller.range, seuil = seuil)
        for i in C:
            if (len(self.clusters) > cpt):
                self.clusters[cpt].rectanglitude = i
                cpt = cpt + 1
        """
        return clustersRectanglitude(distVoisin=self.controller.range, valueTresh = seuil)

    def SumshapeIndex(self):

        cpt = []
        for c in self.clusters:
            cpt = cpt + [c.shape_index]
        cpt = np.array(cpt)
        return cpt.sum()

    def MaxshapeIndex(self):
        cpt = []
        for c in self.clusters:
            cpt = cpt + [c.shape_index]
        cpt = np.array(cpt)
        return cpt.max(cpt)


    def setNb_robots(self,nb):
        self.controller = self.controller.withNombre(nb)

    def setTime(self,time):
        self.controller = self.controller.withTime(time)

    def genererRendu(self):
        self.readDatas()
        pos = self.positions
        U = self.concentrations[:,0]
        pos_vert = pos[np.where(U >= self.seuillage_turing_spots)]
        pos_bleu = pos[np.where(U >= self.seuillage_turing_spots-1)]
        pos_rose = pos[np.where( U >= self.seuillage_turing_spots-2)]
        pos_cyan = pos[np.where( U >= self.seuillage_turing_spots-3)]
        pos_noir = pos[np.where(U < self.seuillage_turing_spots - 3)]
        plt.scatter(pos_noir[:,0],pos_noir[:,1],c = "black")
        plt.scatter(pos_bleu[:,0],pos_bleu[:,1], c="blue")
        plt.scatter(pos_cyan[:,0],pos_cyan[:,1], c="cyan")
        plt.scatter(pos_rose[:,0],pos_rose[:,1], c="pink")
        plt.scatter(pos_vert[:,0],pos_vert[:,1], c="green")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()

    def renduTuringSpot(self,restedespoints = True):
        T = self.turing_spots
        pos = self.positions
        if(restedespoints):
            plt.scatter(pos[:,0],pos[:,1], c="black")
        for s in T:
            X = pos[s,0]
            Y = pos[s,1]
            plt.scatter(X,Y,c=np.random.rand(3,).reshape((1,3)))
        plt.show()




    def executeSimulation(self,nb_elements = None,topology = None,time = None):
        if(nb_elements):
            self.nb_robots = nb_elements
            self.controller = self.controller.withNombre(nb_elements)

        if(topology):
            self.topology = topology
            self.controller = self.controller.withTopology(topology)

        if(time):
            self.time = time
            self.controller = self.controller.withTime(time)

        self.controller.run()
        self.readDatas(verbose=False)
"""
Test permettant de vérifier l'extraction de propriétés statiques de l'essaim.
"""
if(__name__=="__main__"):
    print("Début du test de l'extracteur des propriétés de l'essaim sur le chemin : ",os.getcwd())
    os.chdir("../..")
    print("Le simulateur s'execute sur : ",os.getcwd())

    C = swarmDescriptor("morphogenesis")
    C.controller.rez_params()
    C.setTime(1500)
    C.setTopology("random")
    C.setNb_robots(300)
    C.executeSimulation()
    C.shapeIndex()
    C.calculerTuringSpots(seuil=4)
    #C.shapeIndex()
    for i in C.clusters:
        print(i)
    #C.calculerTuringSpots(0)
    #C.genererRendu()
    #C.renduTuringSpot()
    #print("Sommes U et V : ",C.getSommeUV())
    #print("Etates Clusters : " , C.getDonnesCluster(predOrStates=False))
    #print("Predictions Clusters : " , C.getDonnesCluster(predOrStates=True))
    #print("Turing Spots : " , C.calculerTuringSpots(seuil=4))
    #print("Nombre de Turing Spots : " , C.nb_turing_spots)



