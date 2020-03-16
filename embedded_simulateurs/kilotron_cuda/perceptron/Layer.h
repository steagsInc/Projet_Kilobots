#ifndef UNTITLED_HIDDENLAYER_H
#define UNTITLED_HIDDENLAYER_H

#endif //UNTITLED_HIDDENLAYER_H

typedef struct Layer {
    int nb_input;
    int nb_neurons;
    float **weights;
    float **bias;
} Layer;

Layer *new_layer(int nb_input, int nb_neurons);
void display_layer(Layer *layer);
float **compute_layer(Layer *layer,float **input);
void free_layer(Layer *layer);
