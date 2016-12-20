#include <iostream>
#include <cstring>
#include <malloc.h>
#include <time.h>
#include <assert.h>
#include "config.h"
#include "simpleImp.h"
#include "NaiveGPUTranspose.h"
#include "MatrixTransposeShared.h"
#include "MatrixTransposeSolveBankConflicts.h"
#include "MatrixTransposeUnloop.h"
using namespace std;

const int blockx = BLOCK_X; const int blocky = BLOCK_Y;
const int threadx = BLOCK_SIZE; const int thready = BLOCK_SIZE;
int row = blocky * thready;
int column = blockx * threadx;
typedef void(*transFunc)(const int *, int *, const int, const int);

float transpose0();
float transpose1();
float transpose2();
float transpose3();
float transpose4();
float testFunc(transFunc fun, bool);
float countBandWidth(float);

int main(int argc, char * argv[]) {
    float t0 = transpose0();
    float bandwidth0 = countBandWidth(t0);
    cout<<"CPU used: "<<t0<<"ms."<<endl;
    cout<<bandwidth0<<endl;

    t0 = transpose1();
    bandwidth0 = countBandWidth(t0);
    cout<<"CPU used: "<<t0<<"ms."<<endl;
    cout<<bandwidth0<<endl;

    t0 = transpose2();
    bandwidth0 = countBandWidth(t0);
    cout<<"CPU used: "<<t0<<"ms."<<endl;
    cout<<bandwidth0<<endl;

    t0 = transpose3();
    bandwidth0 = countBandWidth(t0);
    cout<<"CPU used: "<<t0<<"ms."<<endl;
    cout<<bandwidth0<<endl;

    t0 = transpose4();
    bandwidth0 = countBandWidth(t0);
    cout<<"CPU used: "<<t0<<"ms."<<endl;
    cout<<bandwidth0<<endl;
    return 0;
}

float transpose0() {
    int values = 0;
    int **source = new int*[row];
    for (int i = 0; i < row; ++i) {
        source[i] = new int[column];
        for (int j = 0; j < column; ++j) {
            source[i][j] = values;
            values++;
        }
    }

    int **dest = new int*[column];
    for (int i=0; i<column; ++i) {
        dest[i] = new int[row];
    }

    //startTIme
    float elapsedTime;
    clock_t start;
    start = clock();
    simpleTransposeMatrix(source, dest, row, column);
    //endTime
    clock_t end;
    end = clock();
    elapsedTime = (float) (end - start) / (float)CLOCKS_PER_SEC;
    elapsedTime *= 1000;


    //verify
    for (int i = 0; i < row; ++i) {
        for(int j = 0; j < column; ++j) {
            assert(source[i][j] == dest[j][i]);
        }
    }

    for (int i=0; i<row; ++i) delete [] source[i];
    delete [] source;
    for (int i=0; i<column; ++i) delete [] dest[i];
    delete [] dest;

    return elapsedTime;

}
float transpose1() {
    transFunc fun = &naiveGPUTranspose;
    return testFunc(fun, false);
}

float transpose2() {
    transFunc fun = &matrixTransposeShared;
    return testFunc(fun, false);
}
float transpose3() {
    transFunc fun = &matrixTransposeSolveBankConflicts;
    return testFunc(fun, false);
}
float transpose4() {
    transFunc fun = &matrixTransposeUnloop;
    return testFunc(fun, true);
}
float testFunc(transFunc fun, bool flag) {
    float elapsedTime = 0;

    int values = 0;
    int *source, *dest;
    int *d_source, *d_dest;
    size_t size = row * column * sizeof(int);

    source = (int *)malloc(size);
    dest = (int *)malloc(size);

    cudaMalloc((void **)&d_source, size);
    cudaMalloc((void **)&d_dest, size);

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            source[i*column+j] = values;
            values++;
        }
    }

    cudaMemcpy(d_source, source, size, cudaMemcpyHostToDevice);

    if (!flag) {
        dim3 threadPerBlock(threadx, thready);
        dim3 numBlocks(blockx, blocky);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        fun<<<numBlocks, threadPerBlock>>>(d_source, d_dest, row, column);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
    }
    else {
        dim3 threadPerBlock(TILE, SIDE);
        dim3 numBlocks(blockx, blocky);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        fun<<<numBlocks, threadPerBlock>>>(d_source, d_dest, row, column);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
    }

    cudaMemcpy(dest, d_dest, size, cudaMemcpyDeviceToHost);

    cudaFree(d_source);
    cudaFree(d_dest);
    free(source);
    free(dest);

    return elapsedTime;
}

float countBandWidth(float elapsedTime) {
    float bandwidth = (BLOCK_SIZE * BLOCK_Y * BLOCK_SIZE * BLOCK_X * sizeof(int) * 2)
        / (elapsedTime/1000 * 1024 * 1024 * 1024);
    return bandwidth;
}
