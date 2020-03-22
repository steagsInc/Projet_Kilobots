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

    Perceptron *perc = new_perceptron_config("NN.txt");

    write_weights(perc,"../weights.txt");

    return 0;
}
