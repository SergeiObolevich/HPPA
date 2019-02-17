#include "matrix.h"

float** matrixInitialize(const int row, const int col) {
	float** matrix = static_cast<float**>(malloc(row * sizeof(float*)));
	for (auto i = 0; i < row; i++)
		matrix[i] = static_cast<float*>(malloc(col * sizeof(float)));
	return matrix;
}

bool matrixCompare(float** firstMatrix, float** secondMatrix) {
	for (auto row = 0; row < MATRIX_SIZE; row++)
		for (auto col = 0; col < MATRIX_SIZE; col++)
			if (firstMatrix[row][col] != secondMatrix[row][col])
				return false;
	return true;
}

void matrixDestroy(float** matrix) {
	for (int i = 0; i < MATRIX_SIZE; i++) {
		free(matrix[i]);
	}
	free(matrix);
}

float** matrixMultiplyWithVectorization(float** firstMatrix, float** secondMatrix, float** resultMatrix) {
	for (int firstMatrixRow = 0; firstMatrixRow < MATRIX_SIZE; firstMatrixRow++) {
		float* resultRow = resultMatrix[firstMatrixRow];
		for (int firstMatrixCol = 0; firstMatrixCol < MATRIX_SIZE; firstMatrixCol++) {
			const float firstMatrixNumber = firstMatrix[firstMatrixRow][firstMatrixCol];
			float* secondMatrixRow = secondMatrix[firstMatrixCol];
			for (int secondMatrixCol = 0; secondMatrixCol < MATRIX_SIZE; secondMatrixCol++)
				resultRow[secondMatrixCol] += firstMatrixNumber * secondMatrixRow[secondMatrixCol];
		}
	}
	return resultMatrix;
}

float** matrixMultiplyWithoutVectorization(float** firstMatrix, float** secondMatrix, float** resultMatrix) {
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
	return resultMatrix;
}

float** matrixMultiplyWithAVX(float** firstMatrix, float** secondMatrix, float** resultMatrix) {
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
	return resultMatrix;
}

float** matrixMultiplyWithCacheOptimization(float** firstMatrix, float** secondMatrix, float** resultMatrix) {
	for (int cachedMatrixRowL3 = 0; cachedMatrixRowL3 < L3_IN_MATRIX; cachedMatrixRowL3++) {
		for (int cachedMatrixColL3 = 0; cachedMatrixColL3 < L3_IN_MATRIX; cachedMatrixColL3++) {
			for (int cachedMatrixRowL2 = 0; cachedMatrixRowL2 < L2_IN_L3; cachedMatrixRowL2++) {
				for (int cachedMatrixColL2 = 0; cachedMatrixColL2 < L2_IN_L3; cachedMatrixColL2++) {
					for (int cachedMatrixRowL1 = 0; cachedMatrixRowL1 < L1_IN_L2; cachedMatrixRowL1++) {
						for (int cachedMatrixColL1 = 0; cachedMatrixColL1 < L1_IN_L2; cachedMatrixColL1++) {
							for (int movingMatrixRow = 0; movingMatrixRow < MATRIX_SIZE; movingMatrixRow++) {
								float* resultRow = resultMatrix[movingMatrixRow];
								const int startIndex = (L3_SIZE * cachedMatrixRowL3) + (L2_SIZE * cachedMatrixRowL2) + (L1_SIZE * cachedMatrixRowL1);
								const int endIndex = startIndex + L1_SIZE;

								for (int movingMatrixCol = startIndex; movingMatrixCol < endIndex; movingMatrixCol++) {
									float movingMatrixMultiplier = firstMatrix[movingMatrixRow][movingMatrixCol];
									float* secondMatrixRow = secondMatrix[movingMatrixCol];
									const int cachedMatrixColStart = (L3_SIZE * cachedMatrixColL3) + (L2_SIZE * cachedMatrixColL2) + (L1_SIZE * cachedMatrixColL1);

									for (int cachedMatrixColIndex = 0; cachedMatrixColIndex < L1_SIZE; cachedMatrixColIndex++) {
										resultRow[cachedMatrixColStart + cachedMatrixColIndex] += movingMatrixMultiplier * secondMatrixRow[cachedMatrixColStart + cachedMatrixColIndex];
									}
								}
							}
						}
					}
				}
			}
		}
	}
	return resultMatrix;
}