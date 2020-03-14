import os
import numpy as np
from Src.controllers.swarmDescriptor import swarmDescriptor


class topologyOptimisation:
    def __init__(self,topologies = "pile",nb = 250):
        self.Topologies = topologies
        self.nb_robots = nb
        self.Swarm = swarmDescriptor("morphogenesis")
        self.pred = []

    def extract_genotype(self,model=None):
        if model is None:
            model = ['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v']
        P = self.Swarm.controller.read_params()
        w = []
        for i in model:
            w.append(P[i])
        w = np.array(w)
        return w

    def put_genotype(self,w, model=None):
        if model is None:
            model = ['A_VAL', 'B_VAL', 'C_VAL', 'D_VAL', 'E_VAL', 'F_VAL', 'G_VAL', 'D_u', 'D_v']
        P = self.Swarm.controller.read_params()
        for i in range(len(model)):
            P[model[i]] = w[i]
        self.Swarm.controller.write_params(P)

    def computeSimulation(self):
        self.Swarm.controller = self.Swarm.controller.withVisiblite(False)\
            .withTime(1500).withNombre(self.nb_robots).\
            withTopology(self.Topologies).run()





if (__name__=="__main__"):
    print("chemin courant : ",os.getcwd())
    os.chdir("..")
    S = topologyOptimisation()
    #TODO : Violon des loss
    for i in range(0,10):
        S.computeSimulation()



