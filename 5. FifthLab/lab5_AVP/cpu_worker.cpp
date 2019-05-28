#include "cpu_worker.h"


unsigned char filterCPU(unsigned char* image_array, int height, int width, int width_size, int filter[3][3], int division_coef) {
	unsigned char pixel = (
		(
			  image_array[(height    )*(width_size + 2) + (width    )] * (filter[0][0])
			+ image_array[(height    )*(width_size + 2)	+ (width + 1)] * (filter[0][1]) 
			+ image_array[(height    )*(width_size + 2) + (width + 2)] * (filter[0][2])
			+ image_array[(height + 1)*(width_size + 2) + (width    )] * (filter[1][0]) 
			+ image_array[(height + 1)*(width_size + 2) + (width + 1)] * (filter[1][1])
			+ image_array[(height + 1)*(width_size + 2) + (width + 2)] * (filter[1][2]) 
			+ image_array[(height + 2)*(width_size + 2) + (width    )] * (filter[2][0])
			+ image_array[(height + 2)*(width_size + 2) + (width + 1)] * (filter[2][1]) 
			+ image_array[(height + 2)*(width_size + 2) + (width + 2)] * (filter[2][2])
		) 
			/ division_coef
		);

	return pixel;
}

Result perform_CPU_worker(Task task)
{
	auto image_matrix = task.image.matrix;

	const auto start = std::chrono::steady_clock::now();

	for(int height = 0; height < task.work_matrix.height; height++)
		for(int width = 0; width < task.work_matrix.width; width++)
			task.work_matrix.matrix[index(height,width,task.work_matrix.width)] = 
				filterCPU(image_matrix.matrix, height, width, task.work_matrix.width, task.filter, task.division_coef);

	const auto end = std::chrono::steady_clock::now();

	const auto time = std::chrono::duration <double, std::milli>(end - start).count();

	const Result result {
		task.work_matrix,
		time,
	};

	return result;
}
