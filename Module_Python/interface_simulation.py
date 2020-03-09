import os
import re
import json
from decimal import *





def lire_params(save=False):

    os.chdir("kilombo/templates/Turing_morphogenesis")
    with open("morphogenesis.c","r") as file1:
        content = file1.read()
        r = re.findall("//Model_Parameter.*//End_Parameters",content,flags=re.DOTALL)
        r = r[0].replace("#define","").replace("//End_Parameters","").replace("//Model_Parameter","").split("\n")
        param_dict = {}
        for i in r[1:-1]:
            t = (i[1:].split(" "))
            param_dict[t[0]] = float(t[1])
        os.chdir("../../..")
        if(save):
            file2 = open("produced/init.json", "w")
            file2.write(json.dumps(param_dict, sort_keys=True))
    return param_dict


def write_params(P):
    print("Chemon courant avant bug : ",os.getcwd())
    lire_params(True)
    params = open("produced/init.json", "r")
    D = json.load(params)
    os.chdir("kilombo/templates/Turing_morphogenesis")
    with open("morphogenesis.c", "r") as file1:
        content = file1.read()
        #print(content)
        for i in D:
            ch = re.findall(i+" .*",content)
            value = Decimal.from_float(P[i]).quantize(Decimal('.00000001'), rounding=ROUND_DOWN)
            content = content.replace(ch[0],i+ " "+str(value))
        os.chdir("../../..")
        f = open("produced/morphogenesis.c","w")
        f.write(content)

def execute_simulation():
    os.chdir("produced")
    os.system("make")
    os.system("./morphogenesis")
    os.chdir("..")

def read_results():
    D = json.load(open("produced/endstate.json"))
    ticks = D["ticks"]
    bots = D["bot_states"]
    return bots,ticks

#execute_simulation()
#print(read_results())
write_params({})