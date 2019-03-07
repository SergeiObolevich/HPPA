#ifndef MATRIX_H
#define MATRIX_H

#include <cstdlib>
#include <ctime>
#include <immintrin.h>
#include <intrin.h>
#include <cstdio>
#include <algorithm>

#define MATRIX_SIZE 1440
#define L3_SIZE 720
#define L2_SIZE 240
#define L1_SIZE 80

const int L3_IN_MATRIX = MATRIX_SIZE / L3_SIZE;
const int L2_IN_L3 = L3_SIZE / L2_SIZE;
const int L1_IN_L2 = L2_SIZE / L1_SIZE;

bool matrixCompare(float** firstMatrix, float** secondMatrix);
void matrixDestroy(float** matrix);

float** matrixInitialize(const int row, const int col);

float** matrixMultiplyWithVectorization(float** firstMatrix, float** secondMatrix, float** resultMatrix);
float** matrixMultiplyWithoutVectorization(float** firstMatrix, float** secondMatrix, float** resultMatrix);
float** matrixMultiplyWithAVX(float** firstMatrix, float** secondMatrix, float** resultMatrix);
float** matrixMultiplyWithCacheOptimization(float** firstMatrix, float** secondMatrix, float** resultMatrix);

#endif MATRIX_H