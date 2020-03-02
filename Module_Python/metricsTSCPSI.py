# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 11:50:56 2020

@author: PC MILO fixe
"""

import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import math

print(cv2.__version__)

def returnUVList() :
    nodes = []
    fp = 'endstate.json'
    with open(fp) as json_file:
        data = json.load(json_file)
        for state in data['bot_states'] :
            u = state['state'].get('u')
            v = state['state'].get('v')
            nodes.append((u, v))
    

def countTuringSpots(show = False, colorTresh = 2, periTresh = 100) :
    nodes = []
    fp = 'endstate.json'
    with open(fp) as json_file:
        data = json.load(json_file)
    
        #print(data)
        for state in data['bot_states'] : #['bot_states']['state']:
            x = state['x_position']
            y = state['y_position']
            u = state['state'].get('u')
            v = state['state'].get('v')
            nodes.append([x, y, u, v])
    
    points = [(i[0], i[1]) for i in nodes]
    
    # generate data values
    val = [i[2] for i in nodes]
    
    # generate Voronoi tessellation
    vor = Voronoi(points)

    fig, ax = plt.subplots()
    
    
    # plot Voronoi diagram, and fill finite regions with color mapped from speed value
    fig = voronoi_plot_2d(vor, show_points=False, show_vertices=False, s=1, line_width = 0)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]
            
            #en nuances de gris :
            #plt.fill(*zip(*polygon), color= (1-val[r]/6, 1-val[r]/6, 1-val[r]/6))
            #en noir et blanc pour trouver les turing spots:
            plt.fill(*zip(*polygon), color= "white" if val[r] < colorTresh else  "black")
            #en jolies couleurs qui marche pas :
            #plt.fill(*zip(*polygon), color= val[r]) 
    
    plt.axis('equal')  
    plt.axis([-800, 800, -800, 800])   
    plt.axis('off')
    fig.savefig("forContour.png")
    
    image = cv2.imread("forContour.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    edged = cv2.Canny(gray, 30, 200) 
    
    
    # Finding Contours 
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #, offset = (100, 10)) 
    if show :
        cv2.imshow('Canny Edges After Contouring', edged) 
        cv2.waitKey(0)
    
    print("Number of Contours found = " + str(len(contours))) 
    
    if show :
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
        
        cv2.imshow('Contours', image) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
    
    turingSpots = []
    for i in range(len(contours)) :
        perimeter = cv2.arcLength(contours[i],True)
        
        #pour empecher de compter un donut de Turing comme deux spots
        
                 
        if perimeter > periTresh:
            nodonut = True
            for j in range(len(contours)) :
                if cv2.pointPolygonTest(contours[j], (contours[i][0][0][0], contours[i][0][0][1]), False) >= 0 and i < j:
                    nodonut = False
                    print("contour " + str(i) + " refusé car inclu")   
            if nodonut :        
                turingSpots.append(contours[i])
            
    print("Number of Turing Spots found = " + str(len(turingSpots)))  
      
    # Draw all contours 
    # -1 signifies drawing all contours 
    if show :
        for i in range(len(turingSpots)) :
            image = cv2.imread("forContour.png")
            cv2.drawContours(image, [turingSpots[i]], -1, (0, 255, 0), 3) 
              
            cv2.imshow('Turing Spots', image) 
            cv2.waitKey(0) 
            cv2.destroyAllWindows() 
    
    
    #print([i[2] for i in nodes])
    return len(turingSpots)

def shapeIndex(show = False):
    nodes = []
    
    fp = 'endstate.json'
    with open(fp) as json_file:
        data = json.load(json_file)
    
        #print(data)
        for state in data['bot_states'] : #['bot_states']['state']:
            x = state['x_position']
            y = state['y_position']
            u = state['state'].get('u')
            v = state['state'].get('v')
            nodes.append([x, y, u, v])
     
    
    fig, ax = plt.subplots()
    #en noir
    plt.scatter([i[0] for i in nodes], [i[1] for i in nodes], c = ["black"  for i in nodes], s = 120)
    
    plt.axis('equal')    
    plt.axis('off')
    fig.savefig("forContour.png", bbox_inches='tight', pad_inches=0) 
    plt.show()
    
    
    image = cv2.imread("forContour.png")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200) 
    if show :
        cv2.imshow('Canny Edges After Contouring', edged) 
        cv2.waitKey(0)
    
    
    # Finding Contours 
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #, offset = (100, 10)) 
      
    
    
    print("Number of Contours found = " + str(len(contours))) 
    
    if show :
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
        
        cv2.imshow('Contours', image) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows()  
    
    
    truePeri = 0
    trueArea = 0
    for i in range(len(contours)) :
        
        if show :
            image = cv2.imread("forContour.png")
            cv2.drawContours(image, [contours[i]], -1, (0, 255, 0), 3) 
              
            cv2.imshow('Turing Spots', image) 
            cv2.waitKey(0) 
            cv2.destroyAllWindows() 
        
        perimeter = cv2.arcLength(contours[i],True)
        if perimeter > truePeri :
            truePeri = perimeter
            trueArea = cv2.contourArea(contours[0],True)
    
    
    print(truePeri)
    print(trueArea)
    return np.abs(truePeri/(2*np.pi*trueArea))

def norme( u ) :
    """
    prend un vecteur sous la forme d'une paire en entrée
    
    renvoie la norme de ce vecteur
    """
    return np.sqrt(u[0]*u[0] + u[1]*u[1])

def angle(u, v) :
    """
    prend deux vecteur sous la forme de deux paires en entrée
    
    renvoie l'angle entre ces vecteurs (en radians)
    """
    prodScal = u[0]*v[0] + u[1]*v[1]
    if norme(u) == 0 :
        print("u",u)
    if norme(v) == 0 :
        print("v",v)
    cos = prodScal/(norme(u)*norme(v))
    if cos < -1 :
        print("cos trop petit, erreur d'arrondi ? ", cos)
        cos = -1
        
    if cos > 1 :
        print("cos trop grand, erreur d'arrondi ? ", cos)
        cos = 1
    return math.acos(cos)


def shapeCharacterizingPoints(angleTreshold, show = False) :
    
    nodes = []
    
    fp = 'endstate.json'
    with open(fp) as json_file:
        data = json.load(json_file)
    
        #print(data)
        for state in data['bot_states'] : #['bot_states']['state']:
            x = state['x_position']
            y = state['y_position']
            u = state['state'].get('u')
            v = state['state'].get('v')
            nodes.append([x, y, u, v])
     
    
    fig, ax = plt.subplots()
    #en noir
    plt.scatter([i[0] for i in nodes], [i[1] for i in nodes], c = ["black"  for i in nodes], s = 120)
    
    plt.axis('equal')    
    plt.axis('off')
    fig.savefig("forContour.png", bbox_inches='tight', pad_inches=0) 
    plt.show()
    
    
    image = cv2.imread("forContour.png")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200) 
    if show :
        cv2.imshow('Canny Edges After Contouring', edged) 
        cv2.waitKey(0)
    
    
    # Finding Contours 
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #, offset = (100, 10)) 
      
    
    
    print("Number of Contours found = " + str(len(contours))) 
    
    if show :
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3) 
        
        cv2.imshow('Contours', image) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows()  
    
    
    truePeri = cv2.arcLength(contours[0],True)
    trueContour = contours[0]
    
    
    for i in range(len(contours)) :
        if show :
            image = cv2.imread("forContour.png")
            cv2.drawContours(image, [contours[i]], -1, (0, 255, 0), 3) 
              
            cv2.imshow('Turing Spots', image) 
            cv2.waitKey(0) 
            cv2.destroyAllWindows() 
        
        perimeter = cv2.arcLength(contours[i],True)
        if perimeter > truePeri :
            truePeri = perimeter
            trueContour = contours[i]
    
    Q = trueContour
    
    if len(Q) > 3 :
        base = 1
        for k in range(2, len(Q)) :
            if k == len(Q)-1 :
                k = 0
            ang = angle([Q[base][0][0]-Q[k][0][0],Q[base][0][1]-Q[k][0][1]], [Q[k+1][0][0]-Q[k][0][0], Q[k+1][0][1]-Q[k][0][1]] )
            if ang >= angleTreshold :
                Q.remove(k)
            else :
                base = k        
    return Q
    


#print("shapeIndex : " + str(shapeIndex()))
print("countTuringSpots : " + str(countTuringSpots(show = True)))
#print("shapeCharacterizingPoints : " + str(shapeCharacterizingPoints(160)))