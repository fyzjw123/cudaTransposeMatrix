#include "MatrixTransposeShared.h"
#include "config.h"

__global__ void matrixTransposeShared(const int *d_a, int *d_b, const int rows, const int cols) {
    
    __shared__ int mat[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x * BLOCK_SIZE;
    int by = blockIdx.y * BLOCK_SIZE;

    int i = by + threadIdx.y; int j = bx + threadIdx.x;
    int ti = bx + threadIdx.y; int tj = by + threadIdx.x;

    if (i<rows && j<cols)
        mat[threadIdx.x][threadIdx.y] = d_a[i*cols+j];

    __syncthreads();
    if (tj < cols && ti<rows)
        d_b[ti*rows+tj] = mat[threadIdx.y][threadIdx.x];
}
