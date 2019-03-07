#include "matrix.h"

int main(int argc, char *argv[]) {
	float** firstMatrix = matrixInitialize(MATRIX_SIZE, MATRIX_SIZE);
	float** secondMatrix = matrixInitialize(MATRIX_SIZE, MATRIX_SIZE);
	float** firstResultMatrix = matrixInitialize(MATRIX_SIZE, MATRIX_SIZE);
	float** secondResultMatrix = matrixInitialize(MATRIX_SIZE, MATRIX_SIZE);
	float** thirdResultMatrix = matrixInitialize(MATRIX_SIZE, MATRIX_SIZE);
	float** fourthResultMatrix = matrixInitialize(MATRIX_SIZE, MATRIX_SIZE);

	for (auto row = 0; row < MATRIX_SIZE; row++)
		for (auto col = 0; col < MATRIX_SIZE; col++) {
			firstMatrix[row][col] = rand() % 10000;
			secondMatrix[row][col] = rand() % 10000;
			firstResultMatrix[row][col] = 0.0;
			secondResultMatrix[row][col] = 0.0;
			thirdResultMatrix[row][col] = 0.0;
			fourthResultMatrix[row][col] = 0.0;
		}

	auto startTime = __rdtsc();
	firstResultMatrix = matrixMultiplyWithVectorization(firstMatrix, secondMatrix, firstResultMatrix);
	auto endTime = __rdtsc();

	printf("Ticks count with Auto Vectorization: \t\t%llu\n", endTime - startTime);

	startTime = __rdtsc();
	secondResultMatrix = matrixMultiplyWithoutVectorization(firstMatrix, secondMatrix, secondResultMatrix);
	endTime = __rdtsc();

	printf("Ticks count without Auto Vectorization: \t%llu\n", endTime - startTime);

	startTime = __rdtsc();
	thirdResultMatrix = matrixMultiplyWithAVX(firstMatrix, secondMatrix, thirdResultMatrix);
	endTime = __rdtsc();

	printf("Ticks count with AVX command: \t\t\t%llu\n", endTime - startTime);

	startTime = __rdtsc();
	fourthResultMatrix = matrixMultiplyWithCacheOptimization(firstMatrix, secondMatrix, fourthResultMatrix);
	endTime = __rdtsc();

	printf("Ticks count with Cache optimization: \t\t%llu\n", endTime - startTime);

	bool firstCompare = matrixCompare(firstResultMatrix, secondResultMatrix);
	bool secondCompare = matrixCompare(secondResultMatrix, thirdResultMatrix);
	bool thirdCompare = matrixCompare(thirdResultMatrix, fourthResultMatrix);

	printf("Compare: ");

	if (!firstCompare)
		printf("First diff to Second. \n");
	else if (!secondCompare) 
		printf("Second diff to Third. \n");
	else if (!thirdCompare)
		printf("Third diff to Fourth. \n");
	else
		printf("All results is equal. \n");

	matrixDestroy(firstResultMatrix);
	matrixDestroy(secondResultMatrix);
	matrixDestroy(thirdResultMatrix);
	matrixDestroy(fourthResultMatrix);
	matrixDestroy(firstMatrix);
	matrixDestroy(secondMatrix);

	system("pause");
}