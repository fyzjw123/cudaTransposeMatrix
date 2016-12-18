#include <iostream>
#include "simpleImp.h"
using namespace std;

void simpleTransposeMatrix(int **source, int **dest, int raw, int column) {
    for (int i = 0; i < raw; ++i) {
        for (int j = 0; j < column; ++j) {
            dest[j][i] = source[i][j];
        }
    }
}
