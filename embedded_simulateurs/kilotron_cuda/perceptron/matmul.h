#include <cuda_runtime.h>

#ifndef MATMUL_CUH_
#define MATMUL_CUH_ //UNTITLED_MATMUL_H

float custom_rand();
float **mat_init(int n_rows, int n_cols);
float **mat_gen_random(int n_rows, int n_cols);
void mat_destroy(float **m);
float **mat_mul4(int n_a_rows, int n_a_cols, float **a, int n_b_cols, float **b);
float **mat_mul_cuda(int n_a_rows, int n_a_cols, float **a, float **b);
float **mat_add(float **a,float **b,int nb_rows,int nb_cols);
void displayMat(float **m,int n_rows,int n_cols);
void displayList(float *m,int n_rows);

#endif
