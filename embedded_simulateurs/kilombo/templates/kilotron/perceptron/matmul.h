//
// Created by steag on 29/02/2020.
//

#ifndef UNTITLED_MATMUL_H
#define UNTITLED_MATMUL_H

#endif //UNTITLED_MATMUL_H

float custom_rand();
float **mat_init(int n_rows, int n_cols);
float **mat_gen_random(int n_rows, int n_cols);
void mat_destroy(float **m);
float **mat_mul0(int n_a_rows, int n_a_cols, float **a, int n_b_cols, float **b);
float **mat_mul4(int n_a_rows, int n_a_cols, float **a, int n_b_cols, float **b);
float **mat_add(float **a,float **b,int nb_rows,int nb_cols);
void displayMat(float **m,int n_rows,int n_cols);
void displayList(float *m,int n_rows);
