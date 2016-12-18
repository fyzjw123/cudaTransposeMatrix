#ifndef __NAIVEGPU__
#define __NAIVEGPU__
__global__ void naiveGPUTranspose(const int *d_a, int *d_b, const int rows, const int cols);
#endif
