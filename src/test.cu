#include <iostream>
#include <cstring>
#include <malloc.h>
#include <time.h>
#include <assert.h>
#include <iomanip>
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
void statistics(float t0, float t1, float t2, float t3, float t4);

int main(int argc, char * argv[]) {
    float t0 = transpose0();
    cout<<setiosflags(ios::left)<<setw(30)<<"CPU Test"<<"[OK]"<<endl;

    float t1 = transpose1();
    cout<<setiosflags(ios::left)<<setw(30)<<"Naive Transpose Test"<<"[OK]"<<endl;

    float t2 = transpose2();
    cout<<setiosflags(ios::left)<<setw(30)<<"Coalesced Memory Test"<<"[OK]"<<endl;

    float t3 = transpose3();
    cout<<setiosflags(ios::left)<<setw(30)<<"Bank Conflicts Test"<<"[OK]"<<endl;

    float t4 = transpose4();
    cout<<setiosflags(ios::left)<<setw(30)<<"Loop Unrolling Test"<<"[OK]"<<endl;

    cout<<endl;

    statistics(t0, t1, t2, t3, t4);

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

    //verify
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < column; ++j) {
            assert(source[i*column+j] == dest[j*row+i]);
        }
    }

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

void statistics(float t0, float t1, float t2, float t3, float t4) {

    float bandwidth0 = countBandWidth(t0);
    float bandwidth1 = countBandWidth(t1);
    float bandwidth2 = countBandWidth(t2);
    float bandwidth3 = countBandWidth(t3);
    float bandwidth4 = countBandWidth(t4);

    float ssu1 = bandwidth1 / bandwidth0;
    float ssu2 = bandwidth2 / bandwidth1;
    float ssu3 = bandwidth3 / bandwidth2;
    float ssu4 = bandwidth4 / bandwidth3;

    float suvc1 = bandwidth1 / bandwidth0;
    float suvc2 = bandwidth2 / bandwidth0;
    float suvc3 = bandwidth3 / bandwidth0;
    float suvc4 = bandwidth4 / bandwidth0;

    //print Message
    cout<<setiosflags(ios::left)<<setw(30)<<""<<setw(30)<<"Time(ms)"<<setw(30)<<"Bandwidth(GB/s)"
        <<setw(30)<<"Step SpeedUp"<<setw(30)<<"Speed Up vs CPU"<<endl;
    for (int i = 0;i<135;++i) {
        cout<<"-";
    }
    cout<<endl;
    cout<<setiosflags(ios::left)<<setw(30)<<"CPU"<<setw(30)<<t0<<setw(30)<<bandwidth0<<endl;
    cout<<endl;
    cout<<setiosflags(ios::left)<<setw(30)<<"Naive Transpose"<<setw(30)<<t1<<setw(30)<<bandwidth1
        <<setw(30)<<ssu1<<setw(30)<<suvc1<<endl;
    cout<<setiosflags(ios::left)<<setw(30)<<"Coalesced Memory"<<setw(30)<<t2<<setw(30)<<bandwidth2
        <<setw(30)<<ssu2<<setw(30)<<suvc2<<endl;
    cout<<setiosflags(ios::left)<<setw(30)<<"Bank Conflicts"<<setw(30)<<t3<<setw(30)<<bandwidth3
        <<setw(30)<<ssu3<<setw(30)<<suvc3<<endl;
    cout<<setiosflags(ios::left)<<setw(30)<<"Loop Unrolling"<<setw(30)<<t4<<setw(30)<<bandwidth4
        <<setw(30)<<ssu4<<setw(30)<<suvc4<<endl;
}
