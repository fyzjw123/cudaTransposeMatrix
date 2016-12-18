#include "NaiveGPUTranspose.h"

__global__ void naiveGPUTranspose(const int *d_a, int *d_b, const int rows, const int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int index_in = i * cols + j;
    int index_out = j * rows + i;

    if (i < rows && j < cols)
        d_b[index_out] = d_a[index_in];
}
