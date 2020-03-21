import os
import numpy as np
from Src.controllers.swarmDescriptor import swarmDescriptor


class topologyOptimisation:
    def __init__(self,topologies = "pile",nb = 250,visible = False,time = 1500,model = ['D_u', "D_v"]):
        self.Topologies = topologies
        self.nb_robots = nb
        self.visible = visible
        self.time = time
        self.Swarm = swarmDescriptor("morphogenesis")
        self.pred = []
        self.model = model


    def extract_genotype(self):
        P = self.Swarm.controller.read_params()
        w = []
        for i in self.model:
            w.append(P[i])
        w = np.array(w)
        return w

    def put_genotype(self,w):
        P = self.Swarm.controller.read_params()
        for i in range(len(self.model)):
            P[self.model[i]] = w[i]
        self.Swarm.controller.write_params(P)

    def computeSimulation(self):
        self.Swarm.controller = self.Swarm.controller.withVisiblite(self.visible)\
            .withTime(self.time).withNombre(self.nb_robots).\
            withTopology(self.Topologies).run()





if (__name__=="__main__"):
    print("chemin courant : ",os.getcwd())
    os.chdir("../..")
    S = topologyOptimisation()
    #TODO : Violon des loss
    for i in range(0,10):
        S.computeSimulation()



