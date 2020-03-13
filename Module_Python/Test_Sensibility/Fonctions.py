import pandas as pd
import numpy as np
import os
import json
import matplotlib.colors as colors


lock = False

topologies = ["pile","random"]
path_kilotron = "produced"
path = "produced"
distance = lambda x1,y1,x2,y2 : np.sqrt((x1-x2)**2+(y1-y2)**2)

def simulatePerceptron1(topology,number):
    os.chdir(path_kilotron)
    os.system("make")
    setTopology(topology,number)
    os.system("./morphogenesis")
    os.chdir("..")



def simulatePerceptron(topology,number):
    global lock
    if (os.getcwd().split("/")[-1] == "produced"):
        os.chdir("..")
    if (os.getcwd().split("/")[-1] == "Projet_Kilobots"):
        os.chdir("Module_Python")
    if (os.getcwd().split("/")[-1] != "Module_Python"):
        return
    if(lock):
        return
    lock = True
    print("Chemiin : ",os.getcwd())
    os.chdir(path_kilotron)
    os.system("make")
    os.chdir("..")

    setTopology(topology,number)


    os.chdir(path_kilotron)
    os.system("./morphogenesis")
    os.chdir("..")
    print("Chemin en sortie de simulate perceptron : ", os.getcwd())
    lock = False

def generate_neighbours(id,X,neighborhood_max = 55):
    neighbours = []
    for i in range(X.shape[0]):
        if(i != id):
            print(distance(X[i][0],X[i][1],X[id][0],X[id][1]))
            if(distance(X[i][0],X[i][1],X[id][0],X[id][1]) <= neighborhood_max ):
                neighbours.append(i)
    neighbours = np.array(neighbours)
    if(neighbours.shape[0] != 0 ):
        return X[neighbours]
    else:
        return None


def generateSamplesOfSimulations(id, n_sizes, max_size = 250,neighborhood_max = 10):
    sizes = np.random.randint(10,max_size,n_sizes)
    Total = []
    for s in sizes:
        for j in range(len(topologies)):
            simulatePerceptron(topologies[j],s)
            X = readStates()
            N = generate_neighbours(id,X,neighborhood_max)
            if(N.any() != None):
                N = np.hstack( (N,j+np.ones((N.shape[0],1))) )
                Total = Total + list(N)
    return np.array(Total)




def setTopology(topology,number=0):
    if (os.getcwd().split("/")[-1] == "Projet_Kilobots"):
        os.chdir("Module_Python")
    if (os.getcwd().split("/")[-1] != "produced"):
        os.chdir(path)
    print("Chemin dans topology  : ",os.getcwd())
    with open("kilombo.json","r") as fp:
        d = json.load(fp)
        fp.close()
    d["formation"] = topology
    if(number):
        d["nBots"] = int(number)
    with open("kilombo.json", "w") as fp:
        json.dump(d, fp)
    os.chdir("..")


def readStates():
    if (os.getcwd().split("/")[-1] == "Projet_Kilobots"):
        os.chdir("Module_Python")
    if (os.getcwd().split("/")[-1] != "produced"):
        os.chdir(path)
    print("Dans read state : ",os.getcwd())
    with open("endstate.json", "r") as fp:
        d = json.load(fp)["bot_states"]
        X = []
        for bot in d:
            L = []
            L.append(bot["x_position"])
            L.append(bot["y_position"])
            for s in bot['state']:
                L.append(bot['state'][s])
            X.append(L)
    X = np.array(X)
    os.chdir("..")
    print("En sortie de read state : ",os.getcwd())

    return X



if(__name__=="__main__"):
    X = generateSamplesOfSimulations(0,30,neighborhood_max=200)
    print(X)
    X,Y = X[:,2:-1],X[:,-1]
    print(pd.DataFrame(X))
    print(pd.DataFrame(Y))
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    L = MLPClassifier()
    L.fit(X,Y)
    print(cross_val_score(L,X,Y))
    import matplotlib.pyplot as plt
    X_red = X[np.where(Y==1),:]
    X_blue = X[np.where(Y==2),:]
    plt.scatter(X_red[0,:][:,0],X_red[0,:][:,1], marker='^')
    plt.scatter(X_blue[0,:][:,0],X_blue[0,:][:,1],marker='o')
    plt.show()