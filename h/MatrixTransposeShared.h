#ifndef __TRANSPOSESHARED__
#define __TRANSPOSESHARED__
__global__ void matrixTransposeShared(const int *d_a, int *d_b, const int rows, const int cols);
#endif
