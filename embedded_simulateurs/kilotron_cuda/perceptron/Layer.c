//
// Created by steag on 29/02/2020.
//
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Layer.h"
#include "matmul.h"

Layer *new_layer(int nb_input, int nb_neurons){

    Layer *layer = (Layer*)malloc(sizeof(Layer));

    //printf("layer %d x %d\n",nb_input,nb_neurons);

    layer->nb_input=nb_input;
    layer->nb_neurons=nb_neurons;
    layer->weights=mat_gen_random(nb_neurons,nb_input);
    //displayMat(layer->weights,nb_neurons,nb_input);
    layer->bias=mat_gen_random(nb_neurons,1);
    //displayMat(layer->bias,nb_neurons,1);

    return layer;

}

void display_layer(Layer *layer){
    printf("layer %d x %d\n",layer->nb_input,layer->nb_neurons);
    displayMat(layer->weights,layer->nb_neurons,layer->nb_input);
    displayMat(layer->bias,layer->nb_neurons,1);
}

void activation_mat(float **mat,int nb_rows,int nb_cols){

    int i,j;
    for (i = 0; i < nb_rows; ++i)
        for (j = 0; j < nb_cols; ++j)
            mat[i][j] = (float)(1/(1+exp((double)(mat[i][j]))));

}

float **compute_layer(Layer *layer,float **input){

    float **res1 = mat_mul_cuda(layer->nb_neurons,layer->nb_input,layer->weights,input);
    mat_destroy(input);
    float **res2 = mat_add(res1,layer->bias,layer->nb_neurons,1);
    mat_destroy(res1);
    activation_mat(res2,layer->nb_neurons,1);

    return res2;
}

void free_layer(Layer *layer){


  free(layer->weights);
  free(layer->bias);

  free(layer);

}
