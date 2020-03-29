# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 23:16:30 2020

@author: PC MILO fixe
"""
import os

import matplotlib.pyplot as plt

from Src.controllers.swarmDescriptor import swarmDescriptor
from Src.milolib import toussaint


def testEverythingOnAnInstance( points, save = False, affiche = True ) :
    """
    prend un ensemble de points sous la forme d'une liste de paires en entrée et effectue les tests dessus
    affiche les points, l'enveloppe convexe, le diametre, le rectangle minimum et le cercle minimum si plote = Ture
    
    renvoie l'enveloppe sous la forme d'une liste de paires
    """
    if affiche :
        fig, ax = plt.subplots()

    
        plt.scatter([i[0] for i in points], [i[1] for i in points])   
        plt.axis([0, 800, 0, 800])       
        plt.axis('equal')

    env = toussaint.enveloppeConvexe(points)
    
    if affiche :
        x = [i[0] for i in env]
        y = [i[1] for i in env]
        plt.plot(x, y, '-or')
    
    diam = toussaint.calculDiametreOpti(points)
    
    if affiche :
        plt.plot([i[0] for i in diam], [i[1] for i in diam], 'yo-')    
    
    Rect = toussaint.Toussaint(env)
    if affiche :
        plt.plot([i[0] for i in Rect], [i[1] for i in Rect], 'go-')
    
    Circle = toussaint.calculCercleMin(env)
    
    if affiche :
        circPlot = plt.Circle(Circle[0], radius = Circle[1], color = "orange", fill = False)
        ax.add_artist(circPlot)
    
    if save and affiche :
        fig.savefig("everythingPlot.png") 
    
    return env
           
def justToussaintOnAnInstance( points, affiche = True ) : 
    """
    prend un ensemble de points sous la forme d'une liste de paires en entrée et effectue les tests dessus
    affiche les points et l'enveloppe convexe si plote = True
    
    renvoie l'enveloppe sous la forme d'une liste de paires
    """
    if affiche :
        fig, ax = plt.subplots()

    
        plt.scatter([i[0] for i in points], [i[1] for i in points])   
        plt.axis([0, 800, 0, 800])       
        plt.axis('equal')
        
    env = toussaint.enveloppeConvexe(points)
    
    if affiche :
        x = [i[0] for i in env]
        y = [i[1] for i in env]
        plt.plot(x, y, '-or')

    return env


def display_topology(C):
    D = C.data
    print(D)
    points = []
    for i in D:
        points.append((int(i["x_position"]), int(i['y_position'])))
    print("Les points sont : ")
    testEverythingOnAnInstance(points, affiche=True)
    plt.show()


if(__name__=="__main__"):
    print("Début du test de l'extracteur des propriétés de l'essaim sur le chemin : ",os.getcwd())
    os.chdir("../..")
    print("Le simulateur s'execute sur : ",os.getcwd())
    C = swarmDescriptor("morphogenesis")
    C.setTime(1000)
    C.setTopology("random")
    C.setNb_robots(100)
    C.setRange(70)
    C.executeSimulation()
    D= C.data
    print(D)
    points = []
    for i in D:
        points.append((int(i["x_position"]), int(i['y_position'])))
    print("Les points sont : ")
    testEverythingOnAnInstance(points, affiche=True)
    plt.show()



