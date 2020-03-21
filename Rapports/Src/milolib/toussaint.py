# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 16:16:37 2019

@author: PC MILO fixe
"""

import numpy as np
import math

def calculDiametre( points ) :
    """
    prend un ensemble de points sous la forme d'une liste de paires
    
    renvoie le diametre de cet ensemble par la force brute (très couteux, à éviter)
    """
    if len(points) < 3 :
        return
    p = points[0]
    q = points[1]
    distMax = 0
    enveloppe = enveloppeConvexe(points)
    for i in range(len(enveloppe)) :
        for j in range(len(enveloppe)) :
            if(enveloppe[i] != enveloppe[(i+1)%len(enveloppe)]) :
                    
                distance = distancePointLine(enveloppe[j], enveloppe[i], enveloppe[(i+1)%len(enveloppe)])
                
                if distance > distMax :
                    distMax = distance
                    p = enveloppe[j]
                    if(distanceSq(enveloppe[j][0], enveloppe[j][1], enveloppe[i][0], enveloppe[i][1]) > distanceSq(enveloppe[j][0], enveloppe[j][1], enveloppe[(i+1)%len(enveloppe)][0], enveloppe[(i+1)%len(enveloppe)][1])) :
                        q = enveloppe[i]
                    else :
                        q = enveloppe[(i+1)%len(enveloppe)]
                    
    return (p, q)

#calcul avec shamos, il semblerait qu'il trouve également une solution plus fiable (cf jeu de données 17)
def calculDiametreOpti ( points ) :
    """
    prend un ensemble de points sous la forme d'une liste de paires
    
    renvoie le diametre de cet ensemble en calculant l'enveloppe puis les paires
    antipodales
    """
    if len(points) < 3 :
        return
    
    distMax = 0
    
    enveloppe = enveloppeConvexe(points)
    
    antipods = GetAllAntiPodalPairs(enveloppe)
    
    diametre = antipods[0]
    
    for i in antipods :
        if distanceSq(i[0][0], i[0][1], i[1][0], i[1][1]) > distMax :
            distMax = distanceSq(i[0][0], i[0][1], i[1][0], i[1][1])
            diametre = i
        
    return diametre
    
    
#algo de shamos
def GetAllAntiPodalPairs(env) :
    """
    prend une enveloppe convexe sous la forme d'une liste de paires ordonnées
    
    renvoie les paires antipodales sous la forme d'une liste de paires de paires
    """
    
    pairs = []
    n = len(env)
    i0 = 0
    i = 0
    j = (i+1)%n
    #trouve la première paire antipodale avec i et i+1, n calculs dans le pire des cas
    while (Area(env[i%n], env[(i+1)%n], env[j+1]) >= Area(env[i],env[(i+1)%n],env[j])) :
        j = (j+1)%n
        j0 = j
    #tant que l'on n'a pas fait le tour, on fait avancer i et j
    while (j != i0) :
        i = (i+1)%n
        pairs.append( (env[i],env[j]))
        while Area(env[i],env[(i+1)%n],env[(j+1)%n]) > Area(env[i],env[(i+1)%n],env[j]) :
            j=(j+1)%n
            if ((i,j) != (j0,i0)) :
                pairs.append( (env[i],env[j]))
            else :
                return pairs
        if (Area(env[j%n],env[(i+1)%n],env[(j+1)%n]) == Area(env[i],env[(i+1)%n],env[j])) :    
            if ((i,j) != (j0,i0)) :
                pairs.append( (env[i],env[(j+1)%n])) 
            else :
                pairs.append( (env[(i+1)%n],env[j]))
                return pairs
                
            j=(j+1)%n    
            
    return pairs

         
def Area(A, B, C) :
    """
    prend trois points sous la forme de paires en entrée
    
    renvoie l'aire du triangle formé par ces trois points
    """
    if A == B or A == C or B == C :
        return 0
    return distance(A, B)*distancePointLine(C, A, B)/2

def circleArea(circle) :
    """
    prend un cercle sous la forme d'un couple de coordonnées 2D et d'un rayon
    
    renvoie l'aire du cercle
    """
    return math.pi*circle[1]**2

def distanceSq(ax, ay, bx, by) :
    """
    prend deux points sous la forme de paires en entrée
    
    renvoie la distance au carré entre les deux points, utile quand on 
    compare directement des distances
    """
    return (bx-ax)*(bx-ax) + (by-ay)*(by-ay)

def distance(a, b) :
    """
    prend deux points sous la forme de paires en entrée
    
    renvoie la distance entre les deux points
    """
    return np.sqrt((b[0]-a[0])*(b[0]-a[0]) + (b[1]-a[1])*(b[1]-a[1]))

def distancePointLine(X, A, B) :
    """
    prend trois points sous la forme de paires en entrée
    
    renvoie la distance entre le point X et la ligne (A, B)
    """
    if X == A or X == B :
        return 0
    return np.abs(((B[1]-A[1])*X[0] - (B[0]-A[0])*X[1] + B[0]*A[1]-B[1]*A[0]))/ np.sqrt((A[1]-B[1])*(A[1]-B[1]) + (A[0]-B[0])*(A[0]-B[0]))
    

def calculCercleMin( points, epsilon = 10**-6 ) :
    """
    prend un ensemble de points sous la forme d'une liste de paires en entrée
    
    renvoie le cercle minimum de cet ensembel de points par l'algorithme Ritter
    avec le calcul du diamètre optimisé par les paire antipodales
    """
    if len(points) == 0 :
        return
    
    diam = calculDiametreOpti(points)
    P = diam[0]
    Q = diam[1]
    
    C = ((P[0] + Q[0])/2, (P[1] + Q[1])/2)
    
    CERCLE = (C, distance(C, P))
    
    ensembleDesPoints = points.copy()
    
    ensembleDesPoints.remove(P)
    ensembleDesPoints.remove(Q)
    
    while len(ensembleDesPoints) != 0 :
        S = ensembleDesPoints[0]
        
        if distance(S, CERCLE[0]) <= CERCLE[1] :
            ensembleDesPoints.remove(S)
        else :
            Tx = CERCLE[0][0] + CERCLE[1]*(CERCLE[0][0]-S[0])/distance(S, CERCLE[0])
            Ty = CERCLE[0][1] + CERCLE[1]*(CERCLE[0][1]-S[1])/distance(S, CERCLE[0])
            T = (Tx, Ty)
            
            Cprime = ((S[0]+ T[0])/2, (S[1]+ T[1])/2)
            #epsilon est là pour éviter les boucles infinies liées aux erreurs d'arrondis
            CERCLE = (Cprime, distance(Cprime, T) + epsilon)
    return CERCLE

def TriPixel(points) :
    """
    prend un ensemble de points sous la forme d'une liste de paires en entrée
    
    renvoie un ensemble de points sous la forme d'une liste de paires tel 
    que les points conpris entre deux autres points sur une même colonne 
    ont été supprimés
    """

    ymin = 100000000
    ymax =  -100000000
    xmin = 10000000
    xmax = -100000000
    
    for i in points :
        if i[0] < xmin :
            xmin = i[0]
        if i[0] > xmax :
            xmax = i[0]
        if i[1] < ymin :
            ymin = i[1]
        if i[1] > ymax :
            ymax = i[1]
            
    ymins = []
    ymaxs = []
    
    for a in range(xmax-xmin+1) :
        ymins.append(ymax + 1)
        ymaxs.append(ymin - 1)
        
    for i in points :
        if i[1] < ymins[i[0] - xmin] :
            ymins[i[0]-xmin] = i[1]
        if i[1] > ymaxs[i[0] - xmin] :
            ymaxs[i[0]-xmin] = i[1]
        
    
    points2 = []   

    for a in range(len(ymins)) :
        if(ymins[a] != ymax+1  ) :
            points2.append((a+xmin, ymins[a]))
        if(ymaxs[a] != ymin-1 and ymins[a] != ymaxs[a]) :
            points2.append((a+xmin, ymaxs[a]))
            
    return points2

def enveloppeConvexe( points ) :
    """
    prend un ensemble de points sous la forme d'une liste de paires en entrée
    
    renvoie l'enveloppe ordonnée de ce nuage de points après un tri pixel et
    une marche de Jarvis
    """
    if len(points) < 3 :
        return
    enveloppe = []
    points2 = TriPixel(points)
    P = (0, 0)
    absmin = 1000000
    for i in points2 :
        if i[0] < absmin :
            absmin = i[0]
            P = i    
    enveloppe.append(P)
    
    Q = (0, 0)
    for i in points2 :
        onAQ = True
        for j in points2 :
            if((i[0] - P[0])  * (j[1] - P[1]) -  (j[0] - P[0])  * (i[1] - P[1]) < 0) :
                onAQ = False
            if P[0] == i[0] and P[1] == i[1] :
                onAQ = False

        if onAQ :
            Q = i
            break        
    enveloppe.append(Q)
    
    A = P
    B = Q
    while True :
        
        R = (0, 0)
        angleMin = 1000
        for r in points2 :
            if (A[0] == r[0] and A[1] == r[1]) or (B[0] == r[0] and B[1] == r[1]) :
                continue
            else :
                angle1 = angle((B[0]- A[0], B[1]- A[1]), (r[0]- B[0], r[1]- B[1]))
                if angle1 < angleMin :
                    angleMin = angle1
                    R = r            
        A = B
        B = R
        
        if B[0] == P[0] and B[1] == P[1]:
            break
        
        enveloppe.append(R)
    return enveloppe

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

def norme( u ) :
    """
    prend un vecteur sous la forme d'une paire en entrée
    
    renvoie la norme de ce vecteur
    """
    return np.sqrt(u[0]*u[0] + u[1]*u[1])
        
def intersectionPointVector (A, V, B, U) :
    """
    prend deux points (A et B) et deux vecteurs (U et V) sous la forme de deux paires en entrée
    
    renvoie le point d'intersection tel que A + x*V = B + y*U 
    """
    
    if(V[0] != 0 and U[0] != 0) :
        if (V[1]/V[0] - U[1]/U[0]) != 0 :
            x = ((B[1] - B[0] * U[1]/U[0]) - (A[1] - A[0] * V[1]/V[0])) / (V[1]/V[0] - U[1]/U[0])
            y = (A[1] - A[0]*V[1]/V[0]) + x*V[1]/V[0]
        else :
            if U[1] == 0 and V[1] == 0 :
                if A[1] == B[1] :
                    print("erreur : c'est la même ligne horizontale")
                    return
                else :
                    print("erreur : vecteur colinéaire, pas d'intersection")
                    return
    else :
        if U[0] != 0 and V[0] == 0 :
            x = A[0]
            y = (B[1] - B[0]*U[1]/U[0]) + x*U[1]/U[0]

    
        if U[0] == 0 and V[0] != 0 :
            x = B[0]
            y = (A[1] - A[0]*V[1]/V[0]) + x*V[1]/V[0]
            
        if U[0] == 0 and V[0] == 0 :
            if A[0] == B[0] :
                print("erreur : c'est la même ligne verticale")
                return
                
            else :
                print("erreur : vecteurs colinéaire, pas d'intersection")
                return
                
    return x, y


def areaOfConvexPolygon(env) :
    """
    prend un polygone sous la forme d'une liste de paires (coordonnées) en entrée
    
    renvoie la surface du polygone
    """
    n = len(env)
    sum1 = 0
    sum2 = 0
    
    for i in range(len(env)) :
        sum1 = sum1 + env[i][0]*env[(i+1)%n][1]
        sum2 = sum2 + env[i][1]*env[(i+1)%n][0]
    
    return (sum1-sum2)/2



def Toussaint (env) :
    """
    prend une enveloppe convexe ordonnée sous la forme d'une liste de paires (coordonnées) en entrée
    
    renvoie les coordonnées des 4 points du rectangle minimum d'après l'algorithme de Toussaint
    """
    
    n = len(env)
    
    yminVal = 100000000
    ymaxVal =  -100000000
    xminVal = 10000000
    xmaxVal = -100000000
    
    xmin = xmax = ymin = ymax = 0
    
    #détermine les points de départ des pieds à coulisse
    #sous la forme du rectangle minimum aligné à l'axe
    for i in range(len(env)) :

        if env[i][0] < xminVal :
            xminVal = env[i][0]
            xmin = i
        if env[i][0] > xmaxVal :
            xmaxVal = env[i][0]
            xmax = i
        if env[i][1] < yminVal :
            yminVal = env[i][1]
            ymin = i
        if env[i][1] > ymaxVal :
            ymaxVal = env[i][1]
            ymax = i
            
    pi = xmin
    pj = ymax
    pl = ymin
    pk = xmax
    
    vec = (0, -1)
    a = b = c = d = (0, 0)
    
    #on admet que la surface du rectangle minimum est inférieure à ce nombre
    minArea = (xmaxVal-xminVal)*(ymaxVal-yminVal)

    #tant que les pieds à coulisse n'ont pas fait un "quart de tour", on itère :
        # on trouve le plus petit angle pour le prochain côté colinéaire
        # on tourne les pieds à coulisse selon cet angle
        # on calcule le nouveau rectangle
    while pi != (ymin+1)%n :
        #calcul des coordonnées du rectangle
        
        #entre i et j
        aTemp = intersectionPointVector(env[pi], vec, env[pj], (vec[1], -vec[0]))
        #entre j et k
        bTemp = intersectionPointVector(env[pj], (vec[1], -vec[0]), env[pk], (-vec[0], -vec[1]))
        #entre k et l
        cTemp = intersectionPointVector(env[pk], (-vec[0], -vec[1]), env[pl], (-vec[1], vec[0]))
        #entre l et i
        dTemp = intersectionPointVector(env[pl], (-vec[1], vec[0]), env[pi], vec)
        
        areaTemp = areaOfConvexPolygon([dTemp, cTemp, bTemp, aTemp])
             
        if areaTemp <= minArea :
            a = aTemp
            b = bTemp
            c = cTemp
            d = dTemp
            minArea = areaTemp
        
        #calcul de l'angle :
        thetaI = angle(vec, (env[(pi+1)%n][0] - env[pi][0], env[(pi+1)%n][1] - env[pi][1]))
        thetaJ = angle((vec[1], -vec[0]), (env[(pj+1)%n][0] - env[pj][0], env[(pj+1)%n][1] - env[pj][1]))
        thetaK = angle((-vec[0], -vec[1]), (env[(pk+1)%n][0] - env[pk][0], env[(pk+1)%n][1] - env[pk][1]))
        thetaL = angle((-vec[1], vec[0]), (env[(pl+1)%n][0] - env[pl][0], env[(pl+1)%n][1] - env[pl][1]))

        if min([thetaI, thetaJ, thetaK, thetaL]) == thetaI :
            vec = (env[(pi+1)%n][0] - env[pi][0], env[(pi+1)%n][1] - env[pi][1])
            pi = (pi+1)%n
            
        if min([thetaI, thetaJ, thetaK, thetaL]) == thetaJ :
            vec = (-env[(pj+1)%n][1] + env[pj][1] , env[(pj+1)%n][0] - env[pj][0])  
            pj = (pj+1)%n
            
        if min([thetaI, thetaJ, thetaK, thetaL]) == thetaK :
            vec = (env[pk][0] - env[(pk+1)%n][0], env[pk][1] - env[(pk+1)%n][1] ) 
            pk = (pk+1)%n
            
        if min([thetaI, thetaJ, thetaK, thetaL]) == thetaL :
            vec = (env[(pl+1)%n][1] - env[pl][1] , -env[(pl+1)%n][0] + env[pl][0]) 
            pl = (pl+1)%n        
    return d, c, b, a


            

            
            
            
            
            
            
            
            
            
            
            