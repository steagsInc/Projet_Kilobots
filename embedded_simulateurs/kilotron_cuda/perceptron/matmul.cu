#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
extern "C" {
#include "matmul.h"
}
/*******************************************
 * Helper routines for matrix manipulation *
 *******************************************/

float custom_rand(){
    return (float)rand()/(float)(RAND_MAX/1);
}

float **mat_init(int n_rows, int n_cols)
{
	float **m;
	int i;
	m = (float**)malloc(n_rows * sizeof(float*));
	m[0] = (float*)calloc(n_rows * n_cols, sizeof(float));
	for (i = 1; i < n_rows; ++i)
		m[i] = m[i-1] + n_cols;
	return m;
}

void mat_destroy(float **m)
{
	free(m[0]); free(m);
}

float **mat_gen_random(int n_rows, int n_cols)
{
	float **m;
	int i, j;
	m = mat_init(n_rows, n_cols);
	for (i = 0; i < n_rows; ++i)
		for (j = 0; j < n_cols; ++j)
			m[i][j] =custom_rand();
	return m;
}

float **mat_transpose(int n_rows, int n_cols, float *const* a)
{
	int i, j;
	float **m;
	m = mat_init(n_cols, n_rows);
	for (i = 0; i < n_rows; ++i)
		for (j = 0; j < n_cols; ++j)
			m[j][i] = a[i][j];
	return m;
}

float sdot_1(int n, const float *x, const float *y)
{
	int i;
	float s = 0.0f;
	for (i = 0; i < n; ++i) s += x[i] * y[i];
	return s;
}

float sdot_8(int n, const float *x, const float *y)
{
	int i, n8 = n>>3<<3;
	float s, t[8];
	t[0] = t[1] = t[2] = t[3] = t[4] = t[5] = t[6] = t[7] = 0.0f;
	for (i = 0; i < n8; i += 8) {
		t[0] += x[i+0] * y[i+0];
		t[1] += x[i+1] * y[i+1];
		t[2] += x[i+2] * y[i+2];
		t[3] += x[i+3] * y[i+3];
		t[4] += x[i+4] * y[i+4];
		t[5] += x[i+5] * y[i+5];
		t[6] += x[i+6] * y[i+6];
		t[7] += x[i+7] * y[i+7];
	}
	for (s = 0.0f; i < n; ++i) s += x[i] * y[i];
	s += t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
	return s;
}

#ifdef __SSE__
#include <xmmintrin.h>

float sdot_sse(int n, const float *x, const float *y)
{
	int i, n8 = n>>3<<3;
	__m128 vs1, vs2;
	float s, t[4];
	vs1 = _mm_setzero_ps();
	vs2 = _mm_setzero_ps();
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2;
		vx1 = _mm_loadu_ps(&x[i]);
		vx2 = _mm_loadu_ps(&x[i+4]);
		vy1 = _mm_loadu_ps(&y[i]);
		vy2 = _mm_loadu_ps(&y[i+4]);
		vs1 = _mm_add_ps(vs1, _mm_mul_ps(vx1, vy1));
		vs2 = _mm_add_ps(vs2, _mm_mul_ps(vx2, vy2));
	}
	for (s = 0.0f; i < n; ++i) s += x[i] * y[i];
	_mm_storeu_ps(t, vs1);
	s += t[0] + t[1] + t[2] + t[3];
	_mm_storeu_ps(t, vs2);
	s += t[0] + t[1] + t[2] + t[3];
	return s;
}
#endif

/*************************
 * Matrix multiplication *
 *************************/

float **mat_mul4(int n_a_rows, int n_a_cols, float **a, int n_b_cols, float **b)
{
	int i, j, n_b_rows = n_a_cols;
	float **m, **bT;
	m = mat_init(n_a_rows, n_b_cols);
	bT = mat_transpose(n_b_rows, n_b_cols, b);
	for (i = 0; i < n_a_rows; ++i)
		for (j = 0; j < n_b_cols; ++j)
			m[i][j] = sdot_1(n_a_cols, a[i], bT[j]);
	mat_destroy(bT);
	return m;
}

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int n_rows,int nb_entry) {

    int ROW = blockIdx.x*blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < n_rows) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < nb_entry; i++) {
          //printf("%f\n",A[ROW * N + i]);
          //printf("%f\n",B[i * 1 + COL]);
          tmpSum += A[ROW * nb_entry + i] * B[i];
        }
        C[ROW] = tmpSum;
    }
}

float *mat_mul_cuda(float **computeCuda,int n_a_rows, int n_a_cols, float **a, float *b)
{
	int i, j,c = 0;

	for (i = 0; i < n_a_rows; ++i){
    for (j = 0; j < n_a_cols; ++j){
      computeCuda[0][c]=a[i][j];
      c++;
    }
  }

  cudaMemcpy(computeCuda[3], computeCuda[0], n_a_rows*n_a_cols*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(computeCuda[4], b, n_a_rows*n_a_cols*sizeof(float), cudaMemcpyHostToDevice);

  matrixMultiplicationKernel<<<(n_a_rows+1023)/1024, 1024>>>(computeCuda[3], computeCuda[4], computeCuda[5], n_a_rows,n_a_cols);

  cudaMemcpy(computeCuda[2], computeCuda[5], n_a_rows*sizeof(float), cudaMemcpyDeviceToHost);

	return computeCuda[2];
}

/*******************************
 MAT_ADD
 *******************************/

void mat_add(float *a,float **b,int nb_rows){

    int i;
    for (i = 0; i < nb_rows; ++i)
        a[i] = a[i]+b[i][0];
}

//DEBUG

void displayMat(float **m,int n_rows,int n_cols){
    int i, j;
    for (i = 0; i < n_rows; ++i){
        for (j = 0; j < n_cols; ++j)
        {
            printf("%f ", m[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void displayList(float *m,int n_rows){
    int i;
    for (i = 0; i < n_rows; ++i){
        printf("%f ", m[i]);
    }
    printf("\n\n");
}

#ifdef HAVE_CBLAS
#include <cblas.h>

float **mat_mul5(int n_a_rows, int n_a_cols, float *const *a, int n_b_cols, float *const *b)
{
	int i, j, n_b_rows = n_a_cols;
	float **m, **bT;
	m = mat_init(n_a_rows, n_b_cols);
	bT = mat_transpose(n_b_rows, n_b_cols, b);
	for (i = 0; i < n_a_rows; ++i)
		for (j = 0; j < n_b_cols; ++j)
			m[i][j] = cblas_sdot(n_a_cols, a[i], 1, bT[j], 1);
	mat_destroy(bT);
	return m;
}

float **mat_mul6(int n_a_rows, int n_a_cols, float *const *a, int n_b_cols, float *const *b)
{
	float **m, n_b_rows = n_a_cols;
	m = mat_init(n_a_rows, n_b_cols);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_a_rows, n_b_cols, n_a_cols, 1.0f, a[0], n_a_rows, b[0], n_b_rows, 0.0f, m[0], n_a_rows);
	return m;
}
#endif

/*****************
 * Main function *
 *****************/
