import os
import numpy as np
from sklearn.metrics import hinge_loss, log_loss

from Src.controllers.swarmDescriptor import swarmDescriptor

"""
Ce simulateur sert a calculer sur un essaim les differentes loss en effectuant des simulation sur les topologies dans la list
"""

class simulator():
    def __init__(self,topologies = ("circle", "line"),nb = 100):
        self.Topologies = topologies
        self.nb_robots = nb
        self.Swarm = swarmDescriptor("kilotron")
        self.pred = []

    def computeSimulation(self):
        self.Swarm.controller = self.Swarm.controller.withVisiblite(False).withTime(100).withNombre(self.nb_robots)
        self.pred.clear()
        for i in range(len(self.Topologies)):
            self.Swarm.setTopology(topology=self.Topologies[i])
            self.Swarm.executeSimulation()
            self.pred.append(self.Swarm.predictions)


    def getHinge(self):
        cpt = 0
        for i in range(0, len(self.pred)):
            y_p = self.pred[i]
            Z = -np.ones(y_p.shape[0]).astype(int)
            if (i % 2 == 1):
                Z = np.ones(y_p.shape[0]).astype(int)
            P = np.max(y_p, axis=1)
            P[np.where(np.argmax(y_p, axis=1) == 0)] = -P[np.where(np.argmax(y_p, axis=1) == 0)]
            Z  = Z.reshape((Z.shape[0],1))
            P  = P.reshape((P.shape[0],1))
            c = hinge_loss(Z, P)
            cpt = cpt + c
        return cpt

    def getLogLoss(self):
        cpt = 0
        for i in range(0, len(self.pred)):
            y_p = self.pred[i]
            Z = np.hstack((np.zeros((y_p.shape[0], 1)), np.ones((y_p.shape[0], 1))))
            if (i % 2 == 1):
                Z = np.hstack((np.ones((y_p.shape[0], 1)), np.zeros((y_p.shape[0], 1))))
            for k in range(y_p.shape[0]):
                y_p[k, :] = y_p[k, :] / np.sum(y_p[k, :])
            c = log_loss(Z, y_p)
            cpt = cpt + c
        return cpt

    def getPrecision(self):
        cpt = 0
        for i in range(len(self.pred)):
            pred = np.argmax(self.pred[i],axis=1)
            cpt = cpt + np.where(pred == i%2)[0].shape[0]
        cpt = cpt / (len(self.pred)*self.pred[0].shape[0])
        return cpt

    def getLeastSquare(self):
        cpt = 0
        for i in range(0, len(self.pred)):
            y_p = self.pred[i]
            Z = np.zeros(y_p.shape[1])
            # Ensuite j'inscris dessus la bonne prédiction
            Z[i % 2] = 1
            c = np.zeros((1, len(y_p[0])))
            for j in range(y_p.shape[0]):
                # Je compte une pénalité si l'avis du robot donc argmax(p) != de la vraie topologie
                if (np.argmax(y_p[j]) != i % 2):
                    c = c + (y_p[j] - Z) ** 2
            c = np.array(c)
            cpt = cpt + c.sum()
        return cpt

if (__name__=="__main__"):
    print("chemin courant : ",os.getcwd())
    os.chdir("../..")
    S = simulator()
    #TODO : Violon des loss
    for i in range(0,10):
        S.computeSimulation()
        print(S.getPrecision())
