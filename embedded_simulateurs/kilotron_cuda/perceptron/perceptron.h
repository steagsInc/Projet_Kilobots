#ifndef UNTITLED_PERCEPTRON_H
#define UNTITLED_PERCEPTRON_H

#endif //UNTITLED_PERCEPTRON_H

#include "Layer.h"

typedef struct Perceptron {
    int nb_input;
    int nb_layers;
    Layer **layers;

} Perceptron;

Perceptron *new_perceptron(int shape[],int nb_layer);
float *predict(Perceptron *perc,float *input);
void set_layer(Perceptron *perc,int layer_nb,float **weights,float **bias);
void read_weights (const char* file_name);
void write_weights(Perceptron *perc,char* file_name);
void load_weights(Perceptron *perc,char* file_name);
void free_perceptron(Perceptron *perc);
