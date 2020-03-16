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

    layer->computeCuda = (float**)malloc(6*sizeof(float*));

    layer->computeCuda[0] = (float*)malloc(nb_neurons*nb_input*sizeof(float));
    layer->computeCuda[1] = (float*)malloc(nb_neurons*nb_input*sizeof(float));
    layer->computeCuda[2] = (float*)malloc(nb_neurons*sizeof(float));

    cudaMalloc((void**)&layer->computeCuda[3], nb_neurons*nb_input*sizeof(float));
    cudaMalloc((void**)&layer->computeCuda[4], nb_neurons*nb_input*sizeof(float));
    cudaMalloc((void**)&layer->computeCuda[5], nb_neurons*sizeof(float));

    return layer;

}

void display_layer(Layer *layer){
    printf("layer %d x %d\n",layer->nb_input,layer->nb_neurons);
    displayMat(layer->weights,layer->nb_neurons,layer->nb_input);
    displayMat(layer->bias,layer->nb_neurons,1);
}

void activation_mat(float *mat,int nb_rows){

    int i,j;
    for (i = 0; i < nb_rows; ++i)
          mat[i] = (float)(1/(1+exp((double)(mat[i]))));

}

float *compute_layer(Layer *layer,float *input){

    float *res1 = mat_mul_cuda(layer->computeCuda,layer->nb_neurons,layer->nb_input,layer->weights,input);
    mat_add(res1,layer->bias,layer->nb_neurons);
    activation_mat(res1,layer->nb_neurons);

    return res1;
}

void free_layer(Layer *layer){


  free(layer->weights);
  free(layer->bias);

  free(layer->computeCuda[0]);
  free(layer->computeCuda[1]);
  free(layer->computeCuda[2]);
  cudaFree(layer->computeCuda[3]);
  cudaFree(layer->computeCuda[4]);
  cudaFree(layer->computeCuda[5]);

  free(layer->computeCuda);

  free(layer);

}
