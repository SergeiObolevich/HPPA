#include "matrix.h"

float** matrixInitialize(const int row, const int col) {
	float** matrix = static_cast<float**>(malloc(row * sizeof(float*)));
	for (auto i = 0; i < row; i++)
		matrix[i] = static_cast<float*>(malloc(col * sizeof(float)));
	return matrix;
}

void matrixDestroy(float** matrix) {
	for (int i = 0; i < MATRIX_SIZE; i++) {
		free(matrix[i]);
	}
	free(matrix);
}


void matrixMultiplyWithVectorization(float** firstMatrix, float** secondMatrix, float** resultMatrix) {
	for (int firstMatrixRow = 0; firstMatrixRow < MATRIX_SIZE; firstMatrixRow++) {
		float* resultRow = resultMatrix[firstMatrixRow];
		for (int firstMatrixCol = 0; firstMatrixCol < MATRIX_SIZE; firstMatrixCol++) {
			const float firstMatrixNumber = firstMatrix[firstMatrixRow][firstMatrixCol];
			float* secondMatrixRow = secondMatrix[firstMatrixCol];

			for (int secondMatrixCol = 0; secondMatrixCol < MATRIX_SIZE; secondMatrixCol++)
				resultRow[secondMatrixCol] += firstMatrixNumber * secondMatrixRow[secondMatrixCol];
		}
	}
}

void matrixMultiplyWithoutVectorization(float** firstMatrix, float** secondMatrix, float** resultMatrix) {
	for (int firstMatrixRow = 0; firstMatrixRow < MATRIX_SIZE; firstMatrixRow++) {
		float* resultRow = resultMatrix[firstMatrixRow];
		for (int firstMatrixCol = 0; firstMatrixCol < MATRIX_SIZE; firstMatrixCol++) {
			const float firstMatrixNumber = firstMatrix[firstMatrixRow][firstMatrixCol];
			float* secondMatrixRow = secondMatrix[firstMatrixCol];

#pragma loop(no_vector)
			for (int secondMatrixCol = 0; secondMatrixCol < MATRIX_SIZE; secondMatrixCol++)
				resultRow[secondMatrixCol] += firstMatrixNumber * secondMatrixRow[secondMatrixCol];
		}
	}
}

void matrixMultiplyWithAVX(float** firstMatrix, float** secondMatrix, float** resultMatrix) {
	for (int firstMatrixRow = 0; firstMatrixRow < MATRIX_SIZE; firstMatrixRow++) {
		float* resultRow = resultMatrix[firstMatrixRow];
		for (int firstMatrixCol = 0; firstMatrixCol < MATRIX_SIZE; firstMatrixCol++) {
			const float firstMatrixNumber = firstMatrix[firstMatrixRow][firstMatrixCol];
			float* secondMatrixRow = secondMatrix[firstMatrixCol];
			for (int secondMatrixCol = 0; secondMatrixCol < MATRIX_SIZE; secondMatrixCol += 32) {
				const __m256 firstMatrixMultiplier = { firstMatrixNumber, firstMatrixNumber, firstMatrixNumber,
					firstMatrixNumber, firstMatrixNumber, firstMatrixNumber, firstMatrixNumber, firstMatrixNumber };

				__m256 firstPrevResult = _mm256_load_ps(resultRow + secondMatrixCol);
				__m256 secondPrevResult = _mm256_load_ps(resultRow + secondMatrixCol + 8);
				__m256 thirdPrevResult = _mm256_load_ps(resultRow + secondMatrixCol + 16);
				__m256 fourthPrevResult = _mm256_load_ps(resultRow + secondMatrixCol + 24);

				__m256 secondMatrixRowOne = _mm256_load_ps(secondMatrixRow + secondMatrixCol);
				__m256 secondMatrixRowTwo = _mm256_load_ps(secondMatrixRow + secondMatrixCol + 8);
				__m256 secondMatrixRowThree = _mm256_load_ps(secondMatrixRow + secondMatrixCol + 16);
				__m256 secondMatrixRowFour = _mm256_load_ps(secondMatrixRow + secondMatrixCol + 24);

				__m256 firstMulitplyResult = _mm256_mul_ps(firstMatrixMultiplier, secondMatrixRowOne);
				__m256 secondMulitplyResult = _mm256_mul_ps(firstMatrixMultiplier, secondMatrixRowTwo);
				__m256 thirdMulitplyResult = _mm256_mul_ps(firstMatrixMultiplier, secondMatrixRowThree);
				__m256 fourthMulitplyResult = _mm256_mul_ps(firstMatrixMultiplier, secondMatrixRowFour);

				__m256 firstSumResult = _mm256_add_ps(firstPrevResult, firstMulitplyResult);
				__m256 secondSumResult = _mm256_add_ps(secondPrevResult, secondMulitplyResult);
				__m256 thirdSumResult = _mm256_add_ps(thirdPrevResult, thirdMulitplyResult);
				__m256 fourthSumResult = _mm256_add_ps(fourthPrevResult, fourthMulitplyResult);

				_mm256_store_ps(resultRow + secondMatrixCol, firstSumResult);
				_mm256_store_ps(resultRow + secondMatrixCol + 8, secondSumResult);
				_mm256_store_ps(resultRow + secondMatrixCol + 16, thirdSumResult);
				_mm256_store_ps(resultRow + secondMatrixCol + 24, fourthSumResult);
			}
		}
	}
}