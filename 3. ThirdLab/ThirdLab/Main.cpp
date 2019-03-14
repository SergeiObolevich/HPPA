#include "matrix.h"

int main(int argc, char *argv[]) {
	float** firstMatrix = matrixInitialize(MATRIX_SIZE, MATRIX_SIZE);
	float** secondMatrix = matrixInitialize(MATRIX_SIZE, MATRIX_SIZE);
	float** firstResultMatrix = matrixInitialize(MATRIX_SIZE, MATRIX_SIZE);
	float** secondResultMatrix = matrixInitialize(MATRIX_SIZE, MATRIX_SIZE);
	float** thirdResultMatrix = matrixInitialize(MATRIX_SIZE, MATRIX_SIZE);

	for (auto row = 0; row < MATRIX_SIZE; row++)
		for (auto col = 0; col < MATRIX_SIZE; col++) {
			firstMatrix[row][col] = rand() % 10000 + 1;
			secondMatrix[row][col] = rand() % 10000 + 1;
			firstResultMatrix[row][col] = 0.0;
			secondResultMatrix[row][col] = 0.0;
			thirdResultMatrix[row][col] = 0.0;
		}

	
	auto startTime = __rdtsc();
	thread first(matrixMultiplyWithVectorization, firstMatrix, secondMatrix, firstResultMatrix);
	auto endTime = __rdtsc();

	printf("Ticks count with Auto Vectorization: \t\t%llu\n", endTime - startTime);

	startTime = __rdtsc();
	thread second(matrixMultiplyWithoutVectorization, firstMatrix, secondMatrix, secondResultMatrix);
	endTime = __rdtsc();

	printf("Ticks count without Auto Vectorization: \t%llu\n", endTime - startTime);

	startTime = __rdtsc();
	thread third(matrixMultiplyWithAVX, firstMatrix, secondMatrix, thirdResultMatrix);
	endTime = __rdtsc();

	printf("Ticks count with AVX command: \t\t\t%llu\n", endTime - startTime);

	first.detach();
	second.detach();
	third.detach();
	
	
	/*
	auto startTime = __rdtsc();
	matrixMultiplyWithVectorization(firstMatrix, secondMatrix, firstResultMatrix);
	auto endTime = __rdtsc();

	printf("Ticks count with Auto Vectorization: \t\t%llu\n", endTime - startTime);

	startTime = __rdtsc();
	matrixMultiplyWithoutVectorization(firstMatrix, secondMatrix, secondResultMatrix);
	endTime = __rdtsc();

	printf("Ticks count without Auto Vectorization: \t%llu\n", endTime - startTime);

	startTime = __rdtsc();
	matrixMultiplyWithAVX(firstMatrix, secondMatrix, thirdResultMatrix);
	endTime = __rdtsc();

	printf("Ticks count with AVX command: \t\t\t%llu\n", endTime - startTime);
	*/

	matrixDestroy(firstResultMatrix);
	matrixDestroy(secondResultMatrix);
	matrixDestroy(thirdResultMatrix);
	matrixDestroy(firstMatrix);
	matrixDestroy(secondMatrix);

	system("pause");
}