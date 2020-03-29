import os

import cma
import numpy as np
from sklearn.decomposition import pca

from Src.simulationController.topologyOptimizer import topologyOptimisation
import matplotlib.pyplot as plt



S = topologyOptimisation("pile",nb=150,visible=False,time=500)
model_variables = S.model_variables

def sensibilite_param(param,min=None,max=None,wideness=10,nb_points=20,display=("mean_u","mean_v","sigma_u","sigma_v","turing_spots","shape_index")) :
    S.Swarm.controller.rez_params()
    p_init = S.Swarm.controller.read_params()[param]
    if(min == None):
        min = p_init - wideness*p_init
    if (max == None):
        max = p_init + wideness* p_init
    print("min = ",min," max = ",max," nombre de points = ",nb_points)
    X = np.linspace(min,max,nb_points)
    concentration_moyenne_u = []
    concentration_moyenne_v = []
    concentration_sigma_u = []
    concentration_sgima_v = []
    nb_turing_spots = []
    shape_index = []
    for x in X:
        print(param,"= ",x)
        D = {}
        D[param] = float(x)
        S.Swarm.controller.write_params(D)
        S.Swarm.executeSimulation()
        concentration_moyenne_u.append(np.mean(S.Swarm.concentrations,axis=0)[0])
        concentration_moyenne_v.append(np.mean(S.Swarm.concentrations,axis=0)[1])
        concentration_sigma_u.append(np.std(S.Swarm.concentrations, axis=0)[0])
        concentration_sgima_v.append(np.std(S.Swarm.concentrations, axis=0)[1])
        S.Swarm.calculerTuringSpots(seuil=4)
        nb_turing_spots.append(S.Swarm.nb_turing_spots)
        shape_index.append(np.sum(S.Swarm.shapeIndex()))
    fig = plt.figure()
    for opt in display:
        if(opt == "mean_u"):
            plt.xlabel(param)
            plt.ylabel(opt)
            plt.plot(X,concentration_moyenne_u)
            plt.show()
        elif (opt == "mean_v"):
            plt.xlabel(param)
            plt.ylabel(opt)
            plt.plot(X, concentration_moyenne_v)
            plt.show()
        elif (opt == "sigma_u"):
            plt.xlabel(param)
            plt.ylabel(opt)
            plt.plot(X, concentration_sigma_u)
            plt.show()
        elif (opt == "sigma_v"):
            plt.xlabel(param)
            plt.ylabel(opt)
            plt.plot(X, concentration_sgima_v)
            plt.show()
        elif (opt == "turing_spots"):
            plt.xlabel(param)
            plt.ylabel(opt)
            plt.plot(X, nb_turing_spots)
            plt.show()
        elif (opt == "shape_index"):
            plt.xlabel(param)
            plt.ylabel(opt)
            plt.plot(X, shape_index)
            plt.show()
    return



def sensibilite_many(params,mean=0,sigma=1,nb_points=20,display=("mean_u","mean_v","sigma_u","sigma_v","turing_spots","shape_index")):
    S.Swarm.controller.rez_params()
    p_init = {}
    for v in params:
        p_init[v] = S.Swarm.controller.read_params()[v]
    samples = np.random.normal(np.array(len(params)*[mean]), np.array(len(params)*[sigma]), (nb_points,len(params)))
    concentration_moyenne_u = []
    concentration_moyenne_v = []
    concentration_sigma_u = []
    concentration_sgima_v = []
    nb_turing_spots = []
    shape_index = []
    X =[]
    for x in samples:
        print("perturbation = ", x)
        D = {}
        for i in p_init:
            D[i] = float(p_init[i] + x[params.index(i)])
        S.Swarm.controller.write_params(D)
        S.Swarm.executeSimulation()
        concentration_moyenne_u.append(np.mean(S.Swarm.concentrations, axis=0)[0])
        concentration_moyenne_v.append(np.mean(S.Swarm.concentrations, axis=0)[1])
        concentration_sigma_u.append(np.std(S.Swarm.concentrations, axis=0)[0])
        concentration_sgima_v.append(np.std(S.Swarm.concentrations, axis=0)[1])
        S.Swarm.calculerTuringSpots(seuil=4)
        nb_turing_spots.append(S.Swarm.nb_turing_spots)
        shape_index.append(np.sum(S.Swarm.shapeIndex()))
        X.append([p_init[i] + x[params.index(i)] for i in p_init])
    P = pca.PCA(n_components=1)
    X = P.fit_transform(X)
    tri = lambda M:list(np.sort(np.hstack((X,np.array(M).reshape(-1,1))))[:,1])
    concentration_moyenne_u = tri(concentration_moyenne_u)
    concentration_moyenne_v = tri(concentration_moyenne_v)
    concentration_sigma_u = tri(concentration_sigma_u)
    concentration_sgima_v = tri(concentration_sigma_u)
    nb_turing_spots = tri(nb_turing_spots)
    shape_index = tri(shape_index)
    X = np.sort(X.T)
    X = X.T
    for opt in display:
        if (opt == "mean_u"):
            plt.xlabel(params)
            plt.ylabel(opt)
            plt.plot(X, concentration_moyenne_u)
            plt.show()
        elif (opt == "mean_v"):
            plt.xlabel(params)
            plt.ylabel(opt)
            plt.plot(X, concentration_moyenne_v)
            plt.show()
        elif (opt == "sigma_u"):
            plt.xlabel(params)
            plt.ylabel(opt)
            plt.plot(X, concentration_sigma_u)
            plt.show()
        elif (opt == "sigma_v"):
            plt.xlabel(params)
            plt.ylabel(opt)
            plt.plot(X, concentration_sgima_v)
            plt.show()
        elif (opt == "turing_spots"):
            plt.xlabel(params)
            plt.ylabel(opt)
            plt.plot(X, nb_turing_spots)
            plt.show()
        elif (opt == "shape_index"):
            plt.xlabel(params)
            plt.ylabel(opt)
            plt.plot(X, shape_index)
            plt.show()
    return




if(__name__=="__main__"):
    sensibilite_many(["D_u","D_v"],nb_points=10)
