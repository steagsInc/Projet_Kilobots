import os
import numpy as np
from sklearn.metrics import hinge_loss, log_loss

from Src.controllers.swarmDescriptor import swarmDescriptor


class predictionTuring():
    def __init__(self,topologies = ("circle", "line"),nb = 100,model =('A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v')):
        self.Topologies = topologies
        self.nb_robots = nb
        self.Swarm = swarmDescriptor("kilotron_cuda")
        self.pred = []
        self.model = model
        self.Swarm.controller.read_Weights()

    def computeSimulation(self):
        self.Swarm.controller = self.Swarm.controller.withVisiblite(False).withTime(100).withNombre(self.nb_robots)
        self.pred.clear()
        for i in range(len(self.Topologies)):
            self.Swarm.setTopology(topology=self.Topologies[i])
            self.Swarm.executeSimulation()
            self.pred.append(self.Swarm.predictions)
        #print("En sortie la taille de prédictions est : ",len(self.pred))


    #def getHingeContinuous(self):
    #    cpt = 0
    #    for i in range(0, len(self.pred)):
    #        y_p = self.pred[i]
    #        Z = -np.ones(y_p.shape[0]).astype(int)
    #        if (i % 2 == 1):
    #            Z = np.ones(y_p.shape[0]).astype(int)
    #        P = np.max(y_p, axis=1)
    #        P[np.where(np.argmax(y_p, axis=1) == 0)] = -P[np.where(np.argmax(y_p, axis=1) == 0)]
    #        Z  = Z.reshape((Z.shape[0],1))
    #        P  = P.reshape((P.shape[0],1))
    #        c = hinge_loss(Z, P)
    #        cpt = cpt + c
    #    return cpt

    def getHingeContinuous(self):
        cpt = 0
        for i in range(0, len(self.pred)):
            y_p = self.pred[i]
            Z = -np.ones(y_p.shape[0]).astype(int)
            if (i % 2 == 1):
                Z = np.ones(y_p.shape[0]).astype(int)
            P = np.max(y_p, axis=1)
            P[np.where(np.argmax(y_p, axis=1) == 0)] = -P[np.where(np.argmax(y_p, axis=1) == 0)]
            PZ = -P * Z
            PZ[np.where(PZ < 0)] = 0
            # print("P*Z=: ", PZ)
            c = np.sum(PZ)
            cpt = cpt + c
            Z  = Z.reshape((Z.shape[0],1))
            P  = P.reshape((P.shape[0],1))
            cpt = cpt + c
        return cpt

    def softmax(self,predictions):
        predictions = np.exp(predictions)
        for i in range(predictions.shape[0]):
            predictions[i,:] = predictions[i,:] / np.sum(predictions[i,:])
        return predictions

    def getHingeContinuousEcart(self):
        cpt = 0
        for i in range(0, len(self.pred)):
            y_p = self.pred[i]
            Z = -np.ones(y_p.shape[0]).astype(int)
            if (i % 2 == 1):
                Z = np.ones(y_p.shape[0]).astype(int)
            P = np.max(y_p, axis=1) - np.min(y_p,axis=1)
            P[np.where(np.argmax(y_p, axis=1) == 0)] = -P[np.where(np.argmax(y_p, axis=1) == 0)]
            PZ = -P * Z
            PZ[np.where(PZ < 0)] = 0
            # print("P*Z=: ", PZ)
            c = np.sum(PZ)
            cpt = cpt + c
            Z  = Z.reshape((Z.shape[0],1))
            P  = P.reshape((P.shape[0],1))
            cpt = cpt + c
        return cpt

    def getCrossEntropy(self):
        cpt = 0
        loss = 0
        for i in range(0, len(self.pred)):
            y_p = self.pred[i]
            y_p = self.softmax(y_p)
            Z = -np.zeros(y_p.shape).astype(int)
            Z[:,i%2] = 1
            for i in range(0,y_p.shape[0]):
                cpt = 0
                for j in range(0,y_p.shape[1]):
                    cpt = cpt + Z[i][j]*np.log(y_p[i][j])
                loss = loss - cpt
        #print("Donc on retourne : ",cpt)
        return loss

    def getHinge(self):
        cpt = 0
        for i in range(0, len(self.pred)):
            y_p = self.pred[i]
            Z = -np.ones(y_p.shape[0]).astype(int)
            if (i % 2 == 1):
                Z = np.ones(y_p.shape[0]).astype(int)
            P = np.argmax(y_p, axis=1)
            P[np.where(np.argmax(y_p, axis=1) == 0)] = -1
            #print("P=: ", P)
            #print("Z=: ", Z)
            PZ = -P*Z
            PZ[np.where(PZ < 0)] = 0
            #print("P*Z=: ", PZ)
            c =np.sum(PZ)
            cpt = cpt + c
            Z  = Z.reshape((Z.shape[0],1))
            P  = P.reshape((P.shape[0],1))
            #print("Loss ajoutée ",c)
        #print("Donc on retourne : ",cpt)
        return cpt

    def maxHinge(self):
        cpt = []
        for i in range(0, len(self.pred)):
            y_p = self.pred[i]
            Z = -np.ones(y_p.shape[0]).astype(int)
            if (i % 2 == 1):
                Z = np.ones(y_p.shape[0]).astype(int)
            P = np.argmax(y_p, axis=1)
            P[np.where(np.argmax(y_p, axis=1) == 0)] = -1
            #print("P=: ", P)
            #print("Z=: ", Z)
            PZ = -P*Z
            PZ[np.where(PZ < 0)] = 0
            #print("P*Z=: ", PZ)
            c =np.sum(PZ)
            cpt = cpt + [c]
            Z  = Z.reshape((Z.shape[0],1))
            P  = P.reshape((P.shape[0],1))
        return max(cpt)


    def minHinge(self):
        cpt = []
        for i in range(0, len(self.pred)):
            y_p = self.pred[i]
            Z = -np.ones(y_p.shape[0]).astype(int)
            if (i % 2 == 1):
                Z = np.ones(y_p.shape[0]).astype(int)
            P = np.argmax(y_p, axis=1)
            P[np.where(np.argmax(y_p, axis=1) == 0)] = -1
            #print("P=: ", P)
            #print("Z=: ", Z)
            PZ = -P*Z
            PZ[np.where(PZ < 0)] = 0
            #print("P*Z=: ", PZ)
            c =np.sum(PZ)
            cpt = cpt + [c]
            Z  = Z.reshape((Z.shape[0],1))
            P  = P.reshape((P.shape[0],1))
        return min(cpt)

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

    def getPrecisionOnEach(self):
        cpt = []
        #print("en entrée la taille de prediction : ",len(self.pred))
        for i in range(len(self.pred)):
            try:
                #print("Taille des predictions a l'ittération : ",i," : ",len(self.pred))
                predi = np.argmax(self.pred[i],axis=1)
            except(Exception):
                print("erreur")
                return [0.0,0.0]
            cpt.append(np.where(predi == i%2)[0].shape[0])
        cpt=np.array(cpt)
        cpt = cpt / (self.pred[0].shape[0])
        return cpt

    def getEcartPrecision(self):
        p0 = -np.sum(self.pred[0][:][1])/self.nb_robots +np.sum(self.pred[0][:][0])/self.nb_robots
        p1 = -np.sum(self.pred[1][:][0]) / self.nb_robots + np.sum(self.pred[1][:][1]) / self.nb_robots
        return p0+p1

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

    def extract_model_params(self):
        P = self.Swarm.controller.read_params()
        w = []
        for i in self.model:
            w.append(P[i])
        w = np.array(w)
        return w

    def put_model_params(self,w):
        P = self.Swarm.controller.read_params()
        for i in range(len(self.model)):
            P[self.model[i]] = w[i]
        self.Swarm.controller.write_params(P)

    def put_genotype(self,w):
        self.Swarm.controller.read_Weights()
        self.nb_weights = self.Swarm.controller.weights.shape[0]
        nn_weights = w[:self.nb_weights]
        model_params = w[self.nb_weights:]
        self.Swarm.controller.write_Weights(nn_weights)
        self.put_model_params(model_params)

    def get_genotype(self):
        L= list(self.Swarm.controller.read_Weights())+list(self.extract_model_params())
        return np.array(L)


if (__name__=="__main__"):
    print("chemin courant : ",os.getcwd())
    os.chdir("../..")
    S = predictionTuring(nb=10)

    print(S.get_genotype())