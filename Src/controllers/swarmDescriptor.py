import os
import json

from Src.milolib import topologyCalculs
from Src.milolib.topologyCalculs import countTuringSpotsWithGraph, multiClusterShapeIndex
from Src.controllers.simulationController import simulationController
import matplotlib.pyplot as plt
import numpy as np



class Cluster:
    def __init__(self,ids):
        self.cluster_id = None
        self.nb_element = 0
        self.ids = ids
        self.shape_index = 0

    def __str__(self):
        return str(
            dict(
                id = self.cluster_id,
                elements = self.ids,
                size = self.nb_element,
                shape_index = self.shape_index
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
        #Données construites par les calculs
        self.clusters = []
        self.nb_turing_spots = 0
        self.turing_spots = []
        self.seuillage_turing_spots = 5

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
        cpt = 0
        for i in multiClusterShapeIndex(show=False,distVoisin=self.controller.range):
            self.clusters[cpt].shape_index = i
            cpt = cpt + 1
        return multiClusterShapeIndex(show=False, distVoisin=self.controller.range)



    def setNb_robots(self,nb):
        self.controller = self.controller.withNombre(nb)

    def setTime(self,time):
        self.controller = self.controller.withTime(time)


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
    C = swarmDescriptor("kilotron")
    C.setTime(1000)
    C.setTopology("random")
    C.setNb_robots(100)
    C.setRange(70)
    C.executeSimulation()
    C.shapeIndex()
    #print("Sommes U et V : ",C.getSommeUV())
    #print("Etates Clusters : " , C.getDonnesCluster(predOrStates=False))
    #print("Predictions Clusters : " , C.getDonnesCluster(predOrStates=True))
    #print("Turing Spots : " , C.calculerTuringSpots(seuil=4))
    #print("Nombre de Turing Spots : " , C.nb_turing_spots)



