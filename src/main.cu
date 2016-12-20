#include <iostream>
#include <cstring>
#include <malloc.h>
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

void transpose0();
void transpose1();
void transpose2();
void transpose3();
void transpose4();
void testFunc(transFunc fun, bool);

int main(int argc, char * argv[]) {

    string type;
    if (argc > 1) {
        type = string(argv[1]);
    } else {
        cout<<"Usage: Enter parameter which range from 0 to 4."<<endl;
        cout<<"0:CPU, 1:NaiveGPU, 2:SharedMemory, 3:SolveConflicts, 4:Unroll"<<endl;
        return 0;
    }
    if (type.length() > 1) {
        cout<<"Usage: Enter parameter which range from 0 to 4."<<endl;
        cout<<"0:CPU, 1:NaiveGPU, 2:SharedMemory, 3:SolveConflicts, 4:Unroll"<<endl;
        return 0;
    }
    char t = type[0];
    switch (t) {
        case '0':
            transpose0();
            break;
        case '1':
            transpose1();
            break;
        case '2':
            transpose2();
            break;
        case '3':
            transpose3();
            break;
        case '4':
            transpose4();
            break;
        default:
            cout<<"Usage: Enter parameter which range from 0 to 4."<<endl;
            cout<<"0:CPU, 1:NaiveGPU, 2:SharedMemory, 3:SolveConflicts, 4:Unroll"<<endl;
            return 0;
    }

    return 0;
}

void transpose0() {
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

    simpleTransposeMatrix(source, dest, row, column);

    for (int i=0; i<column; ++i) {
        for (int j=0; j<row; ++j) {
            cout<<dest[i][j]<<" ";
        }
        cout<<endl;
    }

    for (int i=0; i<row; ++i) delete [] source[i];
    delete [] source;
    for (int i=0; i<column; ++i) delete [] dest[i];
    delete [] dest;

}
void transpose1() {
    transFunc fun = &naiveGPUTranspose;
    testFunc(fun, false);
}

void transpose2() {
    transFunc fun = &matrixTransposeShared;
    testFunc(fun, false);
}
void transpose3() {
    transFunc fun = &matrixTransposeSolveBankConflicts;
    testFunc(fun, false);
}
void transpose4() {
    transFunc fun = &matrixTransposeUnloop;
    testFunc(fun, true);
}
void testFunc(transFunc fun, bool flag) {

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
        fun<<<numBlocks, threadPerBlock>>>(d_source, d_dest, row, column);
    }
    else {
        dim3 threadPerBlock(TILE, SIDE);
        dim3 numBlocks(blockx, blocky);
        fun<<<numBlocks, threadPerBlock>>>(d_source, d_dest, row, column);
    }

    cudaMemcpy(dest, d_dest, size, cudaMemcpyDeviceToHost);

    for (int i=0; i < column; ++i) {
        for (int j = 0; j < row; ++j) {
            cout<<dest[i*row+j]<<' ';
        }
        cout<<endl;
    }

    cudaFree(d_source);
    cudaFree(d_dest);
    free(source);
    free(dest);
}
