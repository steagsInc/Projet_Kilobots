"""
@author : Milo

Attention !
Ce code a été écrit par Milo, pour votre santé mentale si vous n'etes pas Milo évitez de le lire, vous pourriez plonger dans la démence la plus totale et tenter de vous sucider.

La rédaction d'un code pouvant exploiter les fonctions de ce scripte a couté la vie a plusieurs personnes, et ce commentaire leur sert de mémoriale pour que leur noms
ne soient jamais oublié et que tout le monde se rappelle que c'est grace a eux qu'on peut utiliser ce code de façon totalement transparente en utilisant des objets :

ABERNATHY, Ralph (M) ARAGON, Louis(M) ARGOULT (M) ARON, Raymond (M) BARCLAY, Eddie (M) BARRAULT, Jean-Louis (M) BON, Frédéric (M) BURNIER, Michel-Antoine (M)
CHEVERNY, Julien (M) CHIRAC, Jacques (M) CLAUDIUS-PETIT, Eugène (M) COLOMBO, Pia (Mme) CREVEL, René (M) DALADIER, Edouard (M)
DE SAINT-JORRE (M) DEBRé, Michel (M) DESCAMPS, Eugène (M) DOSTOëWSKY (M) DYLAN, Bob (M) ESCUDERO, Leny (M) EVARISTE (M) FERNIOT, Jean (M) FLAUBERT, Gustave (M)

REST IN PEACE !

TODO : Encoder ce code pour éviter qu'il soit représenté en UTF-8 afin de le sécuriser en évitant que quelqu'un le lise par inadvertance.
"""

import json
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import math



from Src.controllers.simulationController import simulationController
from Src.milolib import toussaint

"""
Oui ça fonctionne vraiment avec un chemin absolu donc changez le si vous voulez executer ce test.
Science sans conscience n'est que ruine de l'âme - A.Einstein
"""

"""
fp = r"/home/mohamed/PycharmProjects/Kilotron_Projet/embedded_simulateurs/morphogenesis/endstate.json"
"""


#quand c'est milo qui code sur son windows
fp = "/home/mohamed/PycharmProjects/Projet_Kilobots/embedded_simulateurs/morphogenesis/endstate.json"


def returnUVList():
    nodes = []
    with open(fp) as json_file:
        data = json.load(json_file)
        for state in data['bot_states']:
            u = state['state'].get('u')
            v = state['state'].get('v')
            nodes.append((u, v))


# mettre le valueTresh à 0 pour que ça compte les clusters au lieu des turing spots
def countTuringSpotsWithGraph(valueTresh=5, distVoisin=100):
    """
    prend les points ayant une valeur de u au dessus de valueTresh et retourne les clusters d'éléments de distance < distvoisin

    retourne les clusters sous forme de liste de liste de leurs id
    """
    nodes = []
    with open(fp) as json_file:
        data = json.load(json_file)

        # print(data)
        for state in data['bot_states']:  # ['bot_states']['state']:
            id = state['ID']
            x = state['x_position']
            y = state['y_position']
            u = state['state'].get('u')
            v = state['state'].get('v')
            nodes.append([x, y, u, v, id, ])

    # on transforme en graphe, mais on garde que les noeuds avec u >= valueTresh d'abors
    nodes2 = []

    for node in range(len(nodes)):
        if nodes[node][2] >= valueTresh:
            nodes2.append(nodes[node])

    nodes = nodes2
    # graphe sous forme de liste d'adjacence de type [id noeud, noeud voisin, noeud voisin...)
    graph = []
    for node in range(len(nodes)):
        graph.append([nodes[node][4]])
        for node2 in range(len(nodes)):
            if nodes[node][4] != nodes[node2][4] and (nodes[node][0] - nodes[node2][0]) ** 2 + (
                    nodes[node][1] - nodes[node2][1]) ** 2 <= distVoisin ** 2:
                graph[node].append(nodes[node2][4])

    # maintenant on prend un noeud au pif et on retire récursivement tous ses voisins du graphe, on répète jusqu'à vide
    list = []
    closedNodes = set()
    openNodes = set()
    while (len(graph) != 0):
        list.append([graph[0][0]])

        for i in range(1, len(graph[0])):
            openNodes.add(graph[0][i])

        closedNodes.add(graph[0][0])
        graph.pop(0)

        while (len(openNodes) != 0):
            elem = openNodes.pop()
            closedNodes.add(elem)

            for e in graph:
                if e[0] == elem:
                    graph.remove(e)

            list[len(list) - 1].append(elem)

            for i in range(len(graph)):
                for j in range(1, len(graph[i])):
                    if graph[i][j] == elem:
                        # if not closedNodes.__contains__(graph[0][j]) :
                        openNodes.add(graph[i][0])
    return list


def countTuringSpotsWithVoronoi(show=False, colorTresh=2, periTresh=100):
    nodes = []
    with open(fp) as json_file:
        data = json.load(json_file)

        # print(data)
        for state in data['bot_states']:  # ['bot_states']['state']:
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
    fig = voronoi_plot_2d(vor, show_points=False, show_vertices=False, s=1, line_width=0)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]

            # en nuances de gris :
            # plt.fill(*zip(*polygon), color= (1-val[r]/6, 1-val[r]/6, 1-val[r]/6))
            # en noir et blanc pour trouver les turing spots:
            plt.fill(*zip(*polygon), color="white" if val[r] < colorTresh else "black")
            # en jolies couleurs qui marche pas :
            # plt.fill(*zip(*polygon), color= val[r])

    plt.axis('equal')
    plt.axis([-800, 800, -800, 800])
    plt.axis('off')
    fig.savefig("forContour.png")
    plt.close()
    
    image = cv2.imread("forContour.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = gray #cv2.Canny(gray, 30, 200)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # , offset = (100, 10))
    #if show:
        #cv2.imshow('Canny Edges After Contouring', edged)
        #cv2.waitKey(0)

    print("Number of Contours found = " + str(len(contours)))

    if show:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

        cv2.imshow('Contours', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    turingSpots = []
    for i in range(len(contours)):
        perimeter = cv2.arcLength(contours[i], True)

        # pour empecher de compter un donut de Turing comme deux spots

        if perimeter > periTresh:
            nodonut = True
            for j in range(len(contours)):
                if cv2.pointPolygonTest(contours[j], (contours[i][0][0][0], contours[i][0][0][1]),
                                        False) >= 0 and i < j:
                    nodonut = False
                    print("contour " + str(i) + " refusé car inclu")
            if nodonut:
                turingSpots.append(contours[i])

    print("Number of Turing Spots found = " + str(len(turingSpots)))

    # Draw all contours
    # -1 signifies drawing all contours
    if show:
        for i in range(len(turingSpots)):
            image = cv2.imread("forContour.png")
            cv2.drawContours(image, [turingSpots[i]], -1, (0, 255, 0), 3)

            cv2.imshow('Turing Spots', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return len(turingSpots)





def countTuringSpotsAnDonutsWithVoronoi(show=False, colorTresh=2, periTresh=100):
    nodes = []
    with open(fp) as json_file:
        data = json.load(json_file)

        # print(data)
        for state in data['bot_states']:  # ['bot_states']['state']:
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
    fig = voronoi_plot_2d(vor, show_points=False, show_vertices=False, s=1, line_width=0)
    for r in range(len(vor.point_region)):
        region = vor.regions[vor.point_region[r]]
        if not -1 in region:
            polygon = [vor.vertices[i] for i in region]

            # en nuances de gris :
            # plt.fill(*zip(*polygon), color= (1-val[r]/6, 1-val[r]/6, 1-val[r]/6))
            # en noir et blanc pour trouver les turing spots:
            plt.fill(*zip(*polygon), color="white" if val[r] < colorTresh else "black")
            # en jolies couleurs qui marche pas :
            # plt.fill(*zip(*polygon), color= val[r])

    plt.axis('equal')
    plt.axis([-800, 800, -800, 800])
    plt.axis('off')
    fig.savefig("forContour.png")
    plt.close()
    
    image = cv2.imread("forContour.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = gray #cv2.Canny(gray, 30, 200)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # , offset = (100, 10))
    #if show:
    #    cv2.imshow('Canny Edges After Contouring', edged)
    #    cv2.waitKey(0)

    print("Number of Contours found = " + str(len(contours)))

    if show:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

        cv2.imshow('Contours', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    turingSpots = []
    for i in range(len(contours)):
        perimeter = cv2.arcLength(contours[i], True)

        # pour compter compter un donut de Turing comme deux spots

        if perimeter > periTresh :
            turingSpots.append(contours[i])

    print("Number of Turing Spots found = " + str(len(turingSpots)))

    # Draw all contours
    # -1 signifies drawing all contours
    if show:
        for i in range(len(turingSpots)):
            image = cv2.imread("forContour.png")
            cv2.drawContours(image, [turingSpots[i]], -1, (0, 255, 0), 3)

            cv2.imshow('Turing Spots', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return len(turingSpots)





def multiClusterShapeIndex(show=False, periTreshold=100, colorTreshold=0, distVoisin=50):
    """
    retourne les shape index de tous les cluster dans une liste, pas seulement le shape index du cluster au plus grand périmètre
    """
    nodes = []

    with open(fp) as json_file:
        data = json.load(json_file)

        # print(data)
        for state in data['bot_states']:  # ['bot_states']['state']:
            x = state['x_position']
            y = state['y_position']
            u = state['state'].get('u')
            v = state['state'].get('v')
            nodes.append([x, y, u, v])

    fig, ax = plt.subplots()

    A = np.array(nodes)
    x = A[:, 0]
    y = A[:, 1]
    c = A[:, 2]

    ax.axis([min(x) - 100, max(x) + 100, min(y) - 100, max(y) + 100])

    for x, y, c in zip(x, y, c):
        ax.add_artist(Circle(xy=(x, y), radius=distVoisin / 2, color="black" if c >= colorTreshold else "white"))
    plt.axis('off')
    fig.savefig("forContour.png", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()
    
    image = cv2.imread("forContour.png")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # edged = cv2.Canny(gray, 30, 200)

    ret, thresh = cv2.threshold(gray, 127, 255, 1)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations=2)

    if show:
        cv2.imshow('Image after erosion', erosion)
        cv2.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # , offset = (100, 10))


    if show:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

        cv2.imshow('Contours', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # modif à partir d'ici

    SI = []
    for i in range(len(contours)):

        perimeter = cv2.arcLength(contours[i], True)
        area = cv2.contourArea(contours[i], True)
        if perimeter > periTreshold:
            if show:
                image = cv2.imread("forContour.png")
                cv2.drawContours(image, [contours[i]], -1, (0, 255, 0), 3)

                cv2.imshow('Turing Spots', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if ((np.pi * area) != 0):
                SI.append(np.abs(perimeter / (2 * np.sqrt(np.abs(np.pi * area)))))
            else:
                SI.append(100)

    return SI


def shapeIndex(show=False, colorTreshold=0, distVoisin=50):
    nodes = []

    with open(fp) as json_file:
        data = json.load(json_file)

        for state in data['bot_states']:  # ['bot_states']['state']:
            x = state['x_position']
            y = state['y_position']
            u = state['state'].get('u')
            v = state['state'].get('v')
            nodes.append([x, y, u, v])

    fig, ax = plt.subplots()

    A = np.array(nodes)
    x = A[:, 0]
    y = A[:, 1]
    c = A[:, 2]

    ax.axis([min(x) - 100, max(x) + 100, min(y) - 100, max(y) + 100])

    for x, y, c in zip(x, y, c):
        ax.add_artist(Circle(xy=(x, y), radius=distVoisin / 2, color="black" if c >= colorTreshold else "white"))

    plt.axis('off')
    fig.savefig("forContour.png", bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()
    
    image = cv2.imread("forContour.png")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edged = cv2.Canny(gray, 30, 200)

    ret, thresh = cv2.threshold(gray, 127, 255, 1)
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel, iterations=2)

    if show:
        # cv2.imshow('Canny Edges After Contouring', edged)
        cv2.imshow('erosion', erosion)
        cv2.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # , offset = (100, 10))


    if show:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

        cv2.imshow('Contours', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    truePeri = 0
    trueArea = 0
    for i in range(len(contours)):
        perimeter = cv2.arcLength(contours[i], True)
        if perimeter > truePeri:
            truePeri = perimeter
            trueArea = cv2.contourArea(contours[0], True)
            mem = i

    if show:
        image = cv2.imread("forContour.png")
        cv2.drawContours(image, [contours[mem]], -1, (0, 255, 0), 3)

        cv2.imshow('Turing Spots', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if ((2 * np.pi * trueArea) != 0):
        return np.abs(truePeri / (2 * np.pi * trueArea))
    else:
        return 100


def norme(u):
    """
    prend un vecteur sous la forme d'une paire en entrée

    renvoie la norme de ce vecteur
    """
    return np.sqrt(u[0] * u[0] + u[1] * u[1])


def angle(u, v):
    """
    prend deux vecteur sous la forme de deux paires en entrée

    renvoie l'angle entre ces vecteurs (en radians)
    """
    prodScal = u[0] * v[0] + u[1] * v[1]
    cos = prodScal / (norme(u) * norme(v))
    if cos < -1:
        print("cos trop petit, erreur d'arrondi ? ", cos)
        cos = -1

    if cos > 1:
        print("cos trop grand, erreur d'arrondi ? ", cos)
        cos = 1
    return math.acos(cos)


def shapeCharacterizingPoints(angleTreshold, show=False, valueTresh = 0):
    nodes = []

    with open(fp) as json_file:
        data = json.load(json_file)

        for state in data['bot_states']:  # ['bot_states']['state']:
            x = state['x_position']
            y = state['y_position']
            u = state['state'].get('u')
            v = state['state'].get('v')
            nodes.append([x, y, u, v])
           
    nodes2 = []

    for node in range(len(nodes)):
        if nodes[node][2] >= valueTresh:
            nodes2.append(nodes[node])

    nodes = nodes2

    fig, ax = plt.subplots()
    # en noir
    plt.scatter([i[0] for i in nodes], [i[1] for i in nodes], c=["black" for i in nodes], s=120)

    plt.axis('equal')
    plt.axis('off')
    fig.savefig("forContour.png", bbox_inches='tight', pad_inches=0)
    if show :
        plt.show()
    plt.close()
        
    image = cv2.imread("forContour.png")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = gray #cv2.Canny(gray, 30, 200)
    if show:
        cv2.imshow('image :', edged)
        cv2.waitKey(0)

    # Finding Contours
    # Use a copy of the image e.g. edged.copy()
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # , offset = (100, 10))
    if show:
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

        cv2.imshow('Contours', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    truePeri = cv2.arcLength(contours[0], True)
    trueContour = contours[0]

    for i in range(len(contours)):
        if show:
            image = cv2.imread("forContour.png")
            cv2.drawContours(image, [contours[i]], -1, (0, 255, 0), 3)

            cv2.imshow('Turing Spots', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        perimeter = cv2.arcLength(contours[i], True)
        if perimeter > truePeri:
            truePeri = perimeter
            trueContour = contours[i]

    Q = trueContour

    if len(Q) > 3:
        base = 1
        for k in range(2, len(Q)):
            if k == len(Q) - 1:
                k = 0
            ang = angle([Q[base][0][0] - Q[k][0][0], Q[base][0][1] - Q[k][0][1]],
                        [Q[k + 1][0][0] - Q[k][0][0], Q[k + 1][0][1] - Q[k][0][1]])
            if ang >= angleTreshold:
                Q.remove(k)
            else:
                base = k
    return Q


def clustersRectanglitude(valueTresh=0, distVoisin=100) :
    """
    retourne une liste contenant la rectanglitude des cluster : l'aire du polygone convexe/l'aire du rectangle de toussaint
    """
    graph = countTuringSpotsWithGraph(valueTresh, distVoisin)
    nodes = []
    with open(fp) as json_file:
        data = json.load(json_file)
        for state in data['bot_states']:  # ['bot_states']['state']:
            id = state['ID']
            x = state['x_position']
            y = state['y_position']
            u = state['state'].get('u')
            v = state['state'].get('v')
            nodes.append([x, y, u, v, id])

    clusters = {}
    for pack in range(len(graph)):
        clusters[pack] = []

    for node in nodes :
        for pack in range(len(graph)) :
            for i in graph[pack] :
                if node[4] == i :
                    clusters[pack].append((  round(node[0]), round(node[1])   ))

    list = []
    for clusterkey in clusters.keys() :
        cluster = clusters.get(clusterkey)
        if len(cluster) > 3 :
            env = toussaint.enveloppeConvexe(cluster)
            areaOfPolygon = toussaint.areaOfConvexPolygon(env)
            areaOfRect = toussaint.areaOfConvexPolygon(toussaint.Toussaint(env))
            if(areaOfRect == 0):
                continue
            list.append(areaOfPolygon/areaOfRect)
        if len(cluster) == 3 :
            list.append(1/2)
        if len(cluster) <= 2 :
            list.append(0)
    if list == [] :
        list = [0]
    return list



"""
Ceci est un test qui sert a montrer pourquoi il ne faut pas programmer comme ça les enfants ! 
prennez de la drogue c'est moins dangereux pour vous et pour la société wolah.
"""
#TODO : Milo Essaye de modifier ces print afin qu'ils donnent une idée intuitive de ce que fait ton code Stp
if(__name__=="__main__"):
    print("Début du test de simulateur sur le chemin : ",os.getcwd())
    os.chdir("../..")
    print("Le simulateur s'execute sur : ",os.getcwd())
    """
    Bonne pratique de la programmation : 
    """
    C = simulationController("morphogenesis").withTime(150).withTopology("pile").withVisiblite(True).withNombre(15).run()
    """
        Mauvaise pratique de la programmation : 
    """
    print(countTuringSpotsWithGraph())
    print(multiClusterShapeIndex())
    print(clustersRectanglitude())
