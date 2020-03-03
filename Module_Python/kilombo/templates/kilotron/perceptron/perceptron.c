//
// Created by steag on 29/02/2020.
//
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "perceptron.h"
#include "matmul.h"

Perceptron *new_perceptron(int shape[],int nb_layer){

    Perceptron *perc = (Perceptron*)malloc(sizeof(Perceptron));
    perc->nb_input=shape[0];
    perc->nb_layers=nb_layer-1;
    perc->layers=(Layer**)malloc((nb_layer-1)* sizeof(Layer*));
    int i;
    for (i=0;i<nb_layer-1;i++){
        perc->layers[i]=new_layer(shape[i],shape[i+1]);
        assert( perc->layers[i] != NULL );
    }

    return perc;

}

float **predict(Perceptron *perc,float **input){

    int i;
    float **a = input;
    for (i=0;i<perc->nb_layers;i++) {
        a = compute_layer(perc->layers[i], a);
    }

    printf("prediction : ");
    displayMat(a,1,1);

}

void set_layer(Perceptron *perc,int layer_nb,float **weights,float **bias){

    perc->layers[layer_nb]->weights= weights;
    perc->layers[layer_nb]->bias=bias;

}