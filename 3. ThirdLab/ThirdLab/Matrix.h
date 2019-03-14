#ifndef MATRIX_H
#define MATRIX_H

#include <cstdlib>
#include <ctime>
#include <immintrin.h>
#include <intrin.h>
#include <cstdio>
#include <algorithm>
#include <climits>
#include <ctime>
#include <iostream>
#include <fstream>
#include <thread>

using namespace std;

#define MATRIX_SIZE 1000

void matrixDestroy(float** matrix);

float** matrixInitialize(const int row, const int col);

void matrixMultiplyWithVectorization(float** firstMatrix, float** secondMatrix, float** resultMatrix);
void matrixMultiplyWithoutVectorization(float** firstMatrix, float** secondMatrix, float** resultMatrix);
void matrixMultiplyWithAVX(float** firstMatrix, float** secondMatrix, float** resultMatrix);

#endif MATRIX_H