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

    int shape[4] = {3,1};
    float **entry = mat_init(3, 1);
    entry[0][0]=1;
    entry[1][0]=1;
    entry[2][0]=1;
    float **weights = mat_init(1, 3);
    weights[0][0]=2;
    weights[0][1]=3;
    weights[0][2]=4;
    float **bias = mat_init(1, 1);
    bias[0][0]=5;


    Perceptron *perc = new_perceptron(shape,2);

    set_layer(perc, 0,weights,bias);

    printf("input : \n");
    displayMat((float **) entry,3,1);
    predict(perc,(float **) entry);

    return 0;
}