#include <cuda_runtime.h>

#ifndef MATMUL_CUH_
#define MATMUL_CUH_ //UNTITLED_MATMUL_H

float custom_rand();
float **mat_init(int n_rows, int n_cols);
float **mat_gen_random(int n_rows, int n_cols);
void mat_destroy(float **m);
float *mat_mul_cuda(float **computeCuda,int n_a_rows, int n_a_cols, float **a, float *b);
void mat_add(float *a,float **b,int nb_rows);
void displayMat(float **m,int n_rows,int n_cols);
void displayList(float *m,int n_rows);

#endif
