#include "gpu_shared_worker.cuh"

__global__	void gpu_shared_updated_filter(unsigned char * image_origininal, unsigned char * image_result, unsigned int width, unsigned int height, int division_coef)
{
	int current_width = blockIdx.y * blockDim.y + threadIdx.y;
	int current_height = blockIdx.x * blockDim.x + threadIdx.x;


	int filter[3][3] =
	{
		{ 1,-2,1 },{ -2,5,-2 },{ 1,-2,1 }
	};

	__shared__ unsigned char block[2][32];

	block[threadIdx.x][threadIdx.y] = (
		(
			  image_origininal[current_height      *(width + 2) + current_width      ] * (filter[0][0])
			+ image_origininal[(current_height    )*(width + 2) + (current_width + 1)] * (filter[0][1])
			+ image_origininal[(current_height    )*(width + 2) + (current_width + 2)] * (filter[0][2])
			+ image_origininal[(current_height + 1)*(width + 2) + (current_width    )] * (filter[1][0])
			+ image_origininal[(current_height + 1)*(width + 2) + (current_width + 1)] * (filter[1][1])
			+ image_origininal[(current_height + 1)*(width + 2) + (current_width + 2)] * (filter[1][2])
			+ image_origininal[(current_height + 2)*(width + 2) + (current_width    )] * (filter[2][0])
			+ image_origininal[(current_height + 2)*(width + 2) + (current_width + 1)] * (filter[2][1])
			+ image_origininal[(current_height + 2)*(width + 2) + (current_width + 2)] * (filter[2][2])
			)
		/ division_coef
		);

	image_result[current_height * width + current_width] = block[threadIdx.x][threadIdx.y];
}

Result perform_GPU_shared_updated_worker(Task task)
{
	cudaEvent_t start_time, stop_time;
	cudaEventCreate(&start_time);
	cudaEventCreate(&stop_time);

	unsigned char* image_original;
	unsigned char* image_result;

	auto cuda_status =
		cudaMalloc((void**)(&image_original),
		(task.image.matrix.height) * (task.image.matrix.width) * sizeof(unsigned char));

	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		exit(EXIT_FAILURE);
	}

	cuda_status = cudaMemcpy(image_original,
		task.image.matrix.matrix,
		(task.image.matrix.height) * (task.image.matrix.width) *
		sizeof(unsigned char), cudaMemcpyHostToDevice);

	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		exit(EXIT_FAILURE);
	}

	cuda_status =
		cudaMalloc((void**)(&image_result),
		(task.work_matrix.height) * (task.work_matrix.width) * sizeof(unsigned char));

	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		exit(EXIT_FAILURE);
	}

	dim3 block(2, 32);
	dim3 grid;

	grid.x = task.work_matrix.height / block.x;
	if (task.work_matrix.height % block.x != 0)
		grid.x += 1;

	grid.y = task.work_matrix.width / block.y;
	if (task.work_matrix.width % block.y != 0)
		grid.y += 1;

	cudaEventRecord(start_time);
	gpu_shared_updated_filter << <grid, block >> > (image_original, image_result, task.work_matrix.width, task.work_matrix.height, task.division_coef);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_time);
	cudaEventSynchronize(stop_time);

	Result result;
	cudaEventElapsedTime(&result.time, start_time, stop_time);

	cuda_status = cudaMemcpy(task.work_matrix.matrix,
		image_result,
		(task.work_matrix.height) * (task.work_matrix.width) * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		exit(EXIT_FAILURE);
	}

	result.result = task.work_matrix;
	cudaEventElapsedTime(&result.time, start_time, stop_time);
	return result;
}
