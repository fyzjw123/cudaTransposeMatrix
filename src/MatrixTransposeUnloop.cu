#include "MatrixTransposeUnloop.h"
#include "config.h"
__global__ void matrixTransposeUnloop(const int * d_a, int * d_b, const int rows, const int cols) {
    
    __shared__ int mat[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    for (int k = 0; k < TILE; k += SIDE) {
        if (x < rows && y + k < cols)
            mat[threadIdx.y + k][threadIdx.x] = d_a[((y + k) * rows) + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    for (int k = 0; k < TILE; k += SIDE) {
        if (x < cols && y + k < rows) {
            d_b[(y + k) * cols + x] = mat[threadIdx.x][threadIdx.y + k];
        }
    }
}
