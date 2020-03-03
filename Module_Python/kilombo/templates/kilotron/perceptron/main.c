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


    Perceptron *perc = new_perceptron(shape,4);

    printf("prout\n" );

    load_weights(perc,"weights.txt");

    display_layer(perc->layers[0]);
    display_layer(perc->layers[1]);
    display_layer(perc->layers[2]);

    printf("input : \n");
    displayMat((float **) entry,3,1);
    predict(perc,(float **) entry);

    return 0;
}
