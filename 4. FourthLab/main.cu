#include <cstdlib>
#include <ostream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <device_functions.h>
#include "device_launch_parameters.h"

using namespace std;

double transform_matrix_cpu(
	short* first_matrix,
	const int first_matrix_height,
	const int first_matrix_width,
	short* second_matrix,
	const int second_matrix_height,
	const int second_matrix_width) {
	auto start_cpu = chrono::steady_clock::now();
	for (auto i = 0; i < first_matrix_height; i++)
		for (auto j = 0; j < first_matrix_width; j+=4) {
			second_matrix[(i * 4) * second_matrix_width + j / 4] = first_matrix[i * first_matrix_width + j + 3];
			second_matrix[(i * 4 + 1) * second_matrix_width + j / 4] = first_matrix[i * first_matrix_width + j + 2];
			second_matrix[(i * 4 + 2) * second_matrix_width + j / 4] = first_matrix[i * first_matrix_width + j + 1];
			second_matrix[(i * 4 + 3) * second_matrix_width + j / 4] = first_matrix[i * first_matrix_width + j];
		}
	auto end_cpu = chrono::steady_clock::now();
	auto cpu_time = end_cpu - start_cpu;
	return  chrono::duration <double, milli>(cpu_time).count();
}

__global__ void kernelSimpleGpu(
	short* first_matrix,
	const int first_matrix_height,
	const int first_matrix_width,
	short* second_matrix,
	const int second_matrix_height,
	const int second_matrix_width) {
	int width = blockIdx.x * blockDim.x + threadIdx.x;
	int height = blockIdx.y * blockDim.y + threadIdx.y;

	if (width > first_matrix_width || height > first_matrix_height)
		return;

	int offset = width % 4;
	int out_height = height * 4 + 3 - offset;
	int out_width = width / 4;

	second_matrix[out_height * second_matrix_width + out_width] = first_matrix[height * first_matrix_width + width];
}

float transform_matrix_gpu_simple(short* first_matrix, const int first_matrix_height, const int first_matrix_width, 
	short* second_matrix, const int second_matrix_height, const int second_matrix_width) {
	cudaEvent_t startTime;
	cudaEvent_t stopTime;
	short* gpu_first_matrix;
	short* gpu_second_matrix;

	cudaMalloc((void**)&gpu_first_matrix, first_matrix_height * first_matrix_width * sizeof(short));
	cudaMemcpy(gpu_first_matrix, first_matrix, first_matrix_height * first_matrix_width * sizeof(short), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&gpu_second_matrix, second_matrix_height * second_matrix_width * sizeof(short));

	dim3 grid;
	dim3 block(32, 32);

	grid.x = first_matrix_height / block.x;
	if (first_matrix_height % block.x != 0)
		grid.x += 1;

	grid.y = first_matrix_width / block.y;
	if (first_matrix_width % block.y != 0)
		grid.y += 1;

	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);
	cudaEventRecord(startTime);

	kernelSimpleGpu << <grid, block >> > (
		gpu_first_matrix,
		first_matrix_height,
		first_matrix_width,
		gpu_second_matrix,
		second_matrix_height,
		second_matrix_width);

	cudaEventRecord(stopTime);
	cudaEventSynchronize(stopTime);


	float result_time;
	cudaEventElapsedTime(&result_time, startTime, stopTime);
	cudaMemcpy(second_matrix, gpu_second_matrix,
		second_matrix_height * second_matrix_width * sizeof(short),
		cudaMemcpyDeviceToHost);

	return result_time;
}

__global__ void kernelSharedGpu(
	short* first_matrix,
	const int first_matrix_height,
	const int first_matrix_width,
	short* second_matrix,
	const int second_matrix_height,
	const int second_matrix_width) {
	int width = blockIdx.x * blockDim.x + threadIdx.x;
	int height = blockIdx.y * blockDim.y + threadIdx.y;

	if (width > first_matrix_width || height > first_matrix_height)
		return;

	__shared__ unsigned short block[32][32];

	int offset = width % 4;
	int out_height = height * 4 + 3 - offset;
	int out_width = width / 4;

	block[threadIdx.y][threadIdx.x] = first_matrix[height * first_matrix_width + width];
	second_matrix[out_height * second_matrix_width + out_width] = block[threadIdx.y][threadIdx.x];
	block[threadIdx.y][threadIdx.x] = 0;
}

float transform_matrix_gpu_shared(short* first_matrix, const int first_matrix_height, const int first_matrix_width, 
	short* second_matrix, const int second_matrix_height, const int second_matrix_width) {
	cudaEvent_t startTime;
	cudaEvent_t stopTime;
	short* gpu_first_matrix;
	short* gpu_second_matrix;

	cudaMalloc((void**)&gpu_first_matrix, first_matrix_height * first_matrix_width * sizeof(short));
	cudaMemcpy(gpu_first_matrix, first_matrix, first_matrix_height * first_matrix_width * sizeof(short), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&gpu_second_matrix, second_matrix_height * second_matrix_width * sizeof(short));

	dim3 grid;
	dim3 block(32, 32);

	grid.x = first_matrix_height / block.x;
	if (first_matrix_height % block.x != 0)
		grid.x += 1;

	grid.y = first_matrix_width / block.y;
	if (first_matrix_width % block.y != 0)
		grid.y += 1;

	cudaEventCreate(&startTime);
	cudaEventCreate(&stopTime);
	cudaEventRecord(startTime);

	kernelSharedGpu << <grid, block >> > (
		gpu_first_matrix,
		first_matrix_height,
		first_matrix_width,
		gpu_second_matrix,
		second_matrix_height,
		second_matrix_width);

	cudaEventRecord(stopTime);
	cudaEventSynchronize(stopTime);

	float result_time;
	cudaEventElapsedTime(&result_time, startTime, stopTime);
	cudaMemcpy(second_matrix, gpu_second_matrix,
		second_matrix_height * second_matrix_width * sizeof(short),
		cudaMemcpyDeviceToHost);

	return result_time;
}

bool compare_matrix(short* first, short* second, int height, int width) {
	for (auto i = 0; i < height; i++)
		for (auto j = 0; j < width; j++)
			if (first[i * width + j] != second[i * width + j])
				return false;
	return true;
}

short* initialize_matrix(const int height, const int width) {
	const auto matrix = static_cast<short *>(calloc(height * width, sizeof(short)));
	return matrix;
}

void fill_random_matrix(short* matrix, int height, int width) {
	short initializer = 0;
	for (auto i = 0; i < height; i++)
		for (auto j = 0; j < width; j++)
			matrix[i * width + j] = rand() % 8 + 1;
}

void show_matrix(short* matrix, const int height, const int width) {
	for (auto i = 0; i < height; i++) {
		for (auto j = 0; j < width; j++)
			cout << setw(2) << matrix[i * width + j];
		cout << endl;
	}
}

int hard_compare_matrix(short* first, short* second, int height, int width) {
	int count_miss = 0;
	for (auto i = 0; i < height; i++)
		for (auto j = 0; j < width; j++)
			if (first[i * width + j] != second[i * width + j]) {
				count_miss++;
			}
	return  count_miss;
}

int main() {
	int first_matrix_height;
	int first_matrix_width;
	cout << "Matrix height: ";
	cin >> first_matrix_height;
	cout << "Matrix width: ";
	cin >> first_matrix_width;

	const int second_matrix_height = first_matrix_height * 4;
	const int second_matrix_width = first_matrix_width / 4;

	const auto first_matrix = initialize_matrix(first_matrix_height, first_matrix_width);
	auto second_matrix = initialize_matrix(second_matrix_height, second_matrix_width);
	auto third_matrix = initialize_matrix(second_matrix_height, second_matrix_width);
	auto fouth_matrix = initialize_matrix(second_matrix_height, second_matrix_width);

	fill_random_matrix(first_matrix, first_matrix_height, first_matrix_width);

	auto cpu_time = transform_matrix_cpu(
		first_matrix,
		first_matrix_height,
		first_matrix_width,
		second_matrix,
		second_matrix_height,
		second_matrix_width);

	auto gpu_simple_time = transform_matrix_gpu_simple(
		first_matrix,
		first_matrix_height,
		first_matrix_width,
		third_matrix,
		second_matrix_height,
		second_matrix_width);

	auto gpu_shared_time = transform_matrix_gpu_shared(
		first_matrix,
		first_matrix_height,
		first_matrix_width,
		fouth_matrix,
		second_matrix_height,
		second_matrix_width);

	/*show_matrix(first_matrix, first_matrix_height, first_matrix_width);
	cout << endl;
	show_matrix(second_matrix, second_matrix_height, second_matrix_width);
	cout << endl;
	show_matrix(third_matrix, second_matrix_height, second_matrix_width);
	cout << endl;
	show_matrix(fouth_matrix, second_matrix_height, second_matrix_width);*/

	cout << "CPU Time: " << cpu_time << " ms." << endl;
	cout << "GPU simple Time: " << gpu_simple_time << " ms." << endl;
	cout << "GPU shared Time: " << gpu_shared_time << " ms." << endl;

	cout << "Compare CPU and Simple - " << compare_matrix(second_matrix, third_matrix, second_matrix_height, second_matrix_width) << endl;
	cout << "Compare CPU and Shared - " << compare_matrix(second_matrix, fouth_matrix, second_matrix_height, second_matrix_width) << endl;
	cout << "Compare Simple and Shared - " << compare_matrix(third_matrix, fouth_matrix, second_matrix_height, second_matrix_width) << endl;

	int first_hard_compare = hard_compare_matrix(second_matrix, third_matrix, second_matrix_height, second_matrix_width);
	int second_hard_compare = hard_compare_matrix(second_matrix, fouth_matrix, second_matrix_height, second_matrix_width);
	int third_hard_compare = hard_compare_matrix(third_matrix, fouth_matrix, second_matrix_height, second_matrix_width);

	cout << "Result of miss count of first hard compare: " << first_hard_compare << endl;
	cout << "Result of miss count of second hard compare: " << second_hard_compare << endl;
	cout << "Result of miss count of third hard compare: " << third_hard_compare << endl;

	system("pause");
}