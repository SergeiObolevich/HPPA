#include <cstdlib>
#include <iostream>
#include <ostream>
#include "image_helper.h"
#include "task_creator.h"
#include "cpu_worker.h"
#include "gpu_simple_worker.cuh"
#include "gpu_shared_worker.cuh"
#include "gpu_shared_updated_worker.cuh"
#include "gpu_shared_buffer_worker.cuh"
#include "gpu_shared_transactions_worker.cuh"

using namespace std;

void compose(unsigned char* first_matrix, unsigned char* second_matrix, int height, int width) {
	int count_of_miss = 0;
	for (int i = 0; i < height*width; i++)
		if (first_matrix[i] != second_matrix[i]) {
			count_of_miss++;
		}
	cout << "Count of miss : " << count_of_miss << endl;
}

int main() {
	auto image = open_image("..\\images\\original.pgm");
	const auto extended_image = extend_image(image);
	
	auto cpu_task = create_task(extended_image);
	auto gpu_simple_task = create_task(extended_image);
	auto gpu_shared_updated_task = create_task(extended_image);
	auto gpu_shared_transactions_task = create_task(extended_image);

	auto cpu_result = perform_CPU_worker(cpu_task);
	auto gpu_simple_result = perform_GPU_simple_worker(gpu_simple_task);
	auto gpu_shared_updated_result = perform_GPU_shared_updated_worker(gpu_shared_updated_task);
	auto gpu_shared_transactions_result = perform_GPU_shared_transactions_worker(gpu_shared_transactions_task);

	const Image cpu_image{
		cpu_result.result,
		"..\\images\\result\\cpu_result.pgm"
	};
	std::cout << "CPU TIME: " << cpu_result.time << "MS" << std::endl;
	save_image(cpu_image);

	const Image gpu_simple_image{
		gpu_simple_result.result,
		"..\\images\\result\\gpu_simple_result.pgm"
	};
	std::cout << "GPU SIMPLE TIME: " << gpu_simple_result.time << "MS" << std::endl;
	save_image(gpu_simple_image);

	const Image gpu_shared_updated_image{
		gpu_shared_updated_result.result,
		"..\\images\\result\\gpu_shared_result.pgm"
	};
	std::cout << "GPU SHARED TIME: " << gpu_shared_updated_result.time << "MS" << std::endl;
	save_image(gpu_shared_updated_image);

	const Image gpu_shared_transactions_image{
		gpu_shared_transactions_result.result,
		"..\\images\\result\\gpu_shared_transactions_result.pgm"
	};
	std::cout << "GPU SHARED TRANSACTIONS TIME: " << gpu_shared_transactions_result.time << "MS" << std::endl;
	save_image(gpu_shared_transactions_image);

	compose(cpu_result.result.matrix, gpu_simple_result.result.matrix,				gpu_simple_result.result.height, gpu_simple_result.result.width);
	compose(cpu_result.result.matrix, gpu_shared_updated_result.result.matrix,		gpu_simple_result.result.height, gpu_simple_result.result.width);
	compose(cpu_result.result.matrix, gpu_shared_transactions_result.result.matrix, gpu_simple_result.result.height, gpu_simple_result.result.width);

	system("pause");
}