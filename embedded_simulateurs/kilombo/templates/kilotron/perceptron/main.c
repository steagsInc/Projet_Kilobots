#include <stdio.h>
#include "matmul.h"
#include <time.h>
#include <stdlib.h>
#include "perceptron.h"

#define NX 3
#define NH 3

int main()
{
    srand (time (NULL));

    int shape[4] = {3,3,3,1};
    float **entry = mat_init(3, 1);
    entry[0][0]=12;
    entry[1][0]=14;
    entry[2][0]=15;

    Perceptron *perc = new_perceptron(shape,4);

    printf("prout\n" );

    //load_weights(perc,"P:\\Projet_Kilobots\\Module_Python\\kilombo\\templates\\kilotron\\perceptron\\weights.txt");

    display_layer(perc->layers[0]);
    display_layer(perc->layers[1]);
    display_layer(perc->layers[2]);

    printf("input : \n");
    displayMat((float **) entry,3,1);
    predict(perc,(float **) entry);

    return 0;
}
