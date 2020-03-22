import os
import json
import re
from decimal import Decimal, ROUND_DOWN
import numpy as np




"""
Pour utiliser le simulateur il faut l'executer sur la racine du projet, dans le doute, on peut utiliser la méthode repositionner et y spécifier le chemin
absolu du projet.
"""

class simulationController:

    def __init__(self,path):
        self.path_project = None
        self.path = path
        self.parametres = json.load(open("embedded_simulateurs/" + self.path + "/kilombo.json","r"))
        self.time = 500
        self.visibility = True
        self.topology = "random"
        self.nb_robots = 100
        self.range = 70
        self.parametres_model = {}
        self.read_params()
        self.weights = None

    def read_params(self,save=False):
        with open("embedded_simulateurs/" + self.path + "/" + self.path+ ".c", "r") as file1:
            content = file1.read()
            r = re.findall("//Model_Parameter.*//End_Parameters", content, flags=re.DOTALL)
            r = r[0].replace("#define", "").replace("//End_Parameters", "").replace("//Model_Parameter", "").split("\n")
            param_dict = {}
            for i in r[1:-1]:
                t = (i[1:].split(" "))
                param_dict[t[0]] = float(t[1])
            if (save):
                file2 = open("produced/init.json", "w")
                file2.write(json.dumps(param_dict, sort_keys=True))
            self.parametres_model = param_dict
            return self.parametres_model

    def write_params(self,P=None):
        if (self.path == "kilotron"):
            return self
        self.read_params()
        if(P):
            self.parametres_model = P
        with open("embedded_simulateurs/natif/morphogenesis.c", "r") as file1:
            content = file1.read()
            for i in (self.parametres_model):
                ch = re.findall(i + " .*", content)
                value = Decimal.from_float(self.parametres_model[i]).quantize(Decimal('.00000000001'), rounding=ROUND_DOWN)
                content = content.replace(ch[0], i + " " + str(value))
            f = open("embedded_simulateurs/" + self.path + "/" + self.path+ ".c", "w")
            f.write(content)
        return self

    def rez_params(self):
        if(self.path == "kilotron"):
            return self
        print("debug rez : ",os.getcwd())
        with open("embedded_simulateurs/natif/morphogenesis.c", "r") as file1:
            content = file1.read()
            f = open("embedded_simulateurs/" + self.path + "/" + self.path + ".c", "w")
            f.write(content)
        return self

    def read_Weights(self):
        with open("embedded_simulateurs/" + self.path + "/weights.txt", "r") as fp:
            c = fp.readlines()
            w = np.zeros(len(c))
            for i in range(len(c)):
                w[i] = float(c[i].strip())
            self.weights = w
        return w

    def write_Weights(self,w):
        self.weights = w
        with open("embedded_simulateurs/" + self.path + "/weights.txt", "w") as fp:
            st = ""
            for i in self.weights:
                st = st + str(i) + "\n"
            fp.write(st)

    def put_Random_Weights(self):
        w = self.read_Weights()
        r = np.random.random(w.shape[0])*np.random.normal(0,1,w.shape[0])
        self.write_Weights(r)

    def getModelParametres(self):
        return self.parametres_model

    def withTime(self,time):
        self.time = time
        self.parametres["simulationTime"] = time
        return self

    def withTopology(self, topology):
        self.topology = topology
        self.parametres["formation"] = topology
        return self

    def withNombre(self, nb):
        self.nb_robots = nb
        self.parametres["nBots"] = nb
        return self

    def withVisiblite(self,visible):
        self.visibility = visible
        self.parametres["GUI"] = 1 if self.visibility else 0
        return self

    def withRange(self,range):
        self.range = range
        self.parametres["commsRadius"] = range
        return self


    def repositionner(self,project_path):
        self.path_project = project_path
        if(project_path.split("/")[-1] != os.getcwd().split("/")[-1]):
            os.chdir(project_path)

    def run(self):
        if(self.path_project and self.path_project.split("/")[-1] != os.getcwd().split("/")[-1]):
            os.chdir(self.path_project)
        if (self.path != os.getcwd().split("/")[-1]):
            os.chdir("embedded_simulateurs/" + self.path)
        json.dump(self.parametres,open("kilombo.json","w"))
        #os.system("make >error.tmp 2>&1")
        os.system("./"+self.path+" >error.tmp 2>&1")
        if (self.path == os.getcwd().split("/")[-1]):
            os.chdir("../..")
        return self




"""
Ceci est un code permettant de verifier le fonctionnement de ce controller de simulateur.
A noté que son fonctionnement est trés dépendant de l'endroit ou le scripte est appelé.
"""
if(__name__=="__main__"):
    print("Début du test de simulateur sur le chemin : ",os.getcwd())
    os.chdir("../..")
    print("Le simulateur s'execute sur : ",os.getcwd())
    C = simulationController("kilotron_cuda").withTime(300).withTopology("line").withVisiblite(True).withNombre(50).run()
    C.read_Weights()
    C.put_Random_Weights()
    print(C.weights)
