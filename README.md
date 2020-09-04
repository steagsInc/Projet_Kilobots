# KEAP | Kilobot's Evolutionary Algorithms in Python
## Introduction
The work present in this repository is part of a project made for the master A.N.D.R.O.I.D at Sorbonne Université. It uses the kilombo simulator available here : https://github.com/JIC-CSB/kilombo 
and from the kilobot code from the paper *Morphogenesis in robot swarm" available here : https://github.com/Danixk/Turing_morphogenesis

## Objectives
We used evolutionnary algorithms to alter the behaviour of morphogenesis by changing the shape of the swarm

## Contenu : 
In This git you can find :
- **Controllers** : This scripts manages the execution of the kilobots program inside of the simulator, it allow you to configure the environment and get results from the simulation.
- **Prototypage** : This module contains all of the Deap evolutionnary algorithms implemented (CMA-ES, CMA-ES Multi objectif, NSGA2) 
- **Experiences** : This directory contains exemples of the differents expriences made throught this project
- **Dashboard** : Web interface made with dash that allow you to directly observe the results from a simulation
## Setup
You can the scripts needed to run the project from docker with this line : -
```python
docker pull ouagueounimohamed/kilotron_project:lates
```
# KEAP | Kilobot's Evolutionary Algorithms in Python
## Introduction
Le contenu de ce repository a été réalisé dans la cadre du projet du master du master agents distribués, recherche opérationnelle, robotique, interaction et decision de l'université de la Sorbonne ex-upmc.
Ce travail se base en grande partie sur celui réalisé dans ce repo : https://github.com/Danixk/Turing_morphogenesis
Il utilise également le simulateur Kilombo disponible ici : https://github.com/JIC-CSB/kilombo

## Objectifs du projet
Le projet vise a utiliser des algorithmes évolutionnaires afin de controler le phénoméne de morphogenèse permettant a l'essaim de s'organiser en une topologie spécifique.
## Contenu : 
Ce repository contient des module permettant de manipuler Kilombo et ce a plusieurs niveaux : 
- **Controllers** : Ces scriptes permettent d'executer un fichier .c dans un environnement kilombo et de récupérer les résultats de l'executions (positions des agents, retours eventuels...etc) mais également des propriétés de l'essaims a la fin de l'exeuctions (clusters, nombre de turing spots etc...) 
- **Prototypage** : Ce module contient les differents algorithmes évolutionnaires que nous avons prototypés sous Deap (CMA-ES, CMA-ES Multi objectif, NSGA2) 
- **Experiences** : Ce module contient des exemples des différentes expérimentations que nous avons menés, pour se référer a l'intégralité des résultats obtenus vous pouvez consulter notre rapport.
- **Dashboard** : Interface web permettant d'executer une simulation et de visualiser l'évolution de l'essaim en temps réel.
## Deployer notre projet
La pile applicative nécessaire a l'execution des scriptes qui composent notre projet est disponible sur docker et récupérable avec la commande suivante : -
```python
docker pull ouagueounimohamed/kilotron_project:lates
```
