import os,time
import matplotlib.pyplot as plt
import numpy as np

def changeShape(template,NN):
    with open(template+'/'+template+'.c', 'r') as file:
        data = file.readlines()
        data[50] = "#define NN " + str(NN) + "\n"

    with open(template+'/'+template+'.c', 'w') as file:
        file.writelines(data)

def changeShapeFile(template,NN):
    with open(template+'/perceptron/NN.txt', 'r') as file:
        data = file.readlines()
        data[4] = str(NN) + "\n"

    with open(template+'/perceptron/NN.txt', 'w') as file:
        file.writelines(data)

    os.chdir(template + "/perceptron")
    os.system('./refreshWeights')
    os.chdir("../..")

def run(template):

    os.chdir(template)
    os.system("make >error.tmp 2>&1")
    start = time.time()
    os.system("./"+template+" >error.tmp 2>&1")
    end = time.time() - start
    os.chdir("..")

    return end

def mesure(max,step):

    t=[]
    t_cuda=[]

    for i in range(10,max+1,step):
        changeShape("kilotron",i)
        t.append(run("kilotron"))

        changeShapeFile("kilotron_cuda", i)
        t_cuda.append(run("kilotron_cuda"))
        print(i)

    n = np.array(range(10,max+1,step))

    plt.plot(n, np.array(t))
    plt.plot(n, np.array(t_cuda))
    plt.show()


mesure(100,10)