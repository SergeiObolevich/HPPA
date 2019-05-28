#include "gpu_shared_transactions_worker.cuh"

const int BLOCKDIM_Y = 8;
const int BLOCKDIM_X = 32;
const int ELEMENT_IN_THREAD_WIDTH = 4;
const int BLOCK_ELEMENT_X = BLOCKDIM_X * ELEMENT_IN_THREAD_WIDTH;

__device__ unsigned char get_element(unsigned char * array, unsigned int height, unsigned int width, unsigned int width_size, size_t pitch)
{
	return (array + height * pitch)[width];
}

__device__ unsigned int get_index(unsigned int height, unsigned int width, unsigned int width_size, size_t pitch)
{
	return height * pitch + width;
}

__global__	void gpu_shared_transactions_filter(
	unsigned char * original_extended_image,
	unsigned int original_width,
	size_t original_pitch,
	unsigned char * image_result,
	unsigned int result_width,
	unsigned int result_height,
	size_t result_pitch,
	unsigned int devision_coefficent
)
{
	int result_current_width = (blockIdx.x * BLOCK_ELEMENT_X) + (threadIdx.x * ELEMENT_IN_THREAD_WIDTH);
	int result_current_height = (blockDim.y * blockIdx.y) + threadIdx.y;

	int original_current_width = result_current_width + 1;
	int original_current_height = result_current_height + 1;

	__shared__ unsigned char temp_image[BLOCKDIM_Y + 2][BLOCK_ELEMENT_X + 2];

	int filter[3][3] =
	{
		{ 1,-2,1 },{ -2,5,-2 },{ 1,-2,1 }
	};

	temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1] = 
			get_element(
				original_extended_image,
				original_current_height, 
				original_current_width,
				original_width,
				original_pitch
			);

	temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2] = 
		get_element(
			original_extended_image,
			original_current_height,
			original_current_width + 1,
			original_width,
			original_pitch
		);

	temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3] = 
		get_element(
			original_extended_image,
			original_current_height,
			original_current_width + 2,
			original_width,
			original_pitch
		);

	temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 4] = 
		get_element(
			original_extended_image,
			original_current_height,
			original_current_width + 3,
			original_width,
			original_pitch
		);

	{
		if (threadIdx.x == 0 && threadIdx.y == 0)
		{
			temp_image[0][0] = 
				get_element(
					original_extended_image,
					original_current_height - 1, 
					original_current_width - 1,
					original_width,
					original_pitch
				);
		}

		if (threadIdx.x == BLOCKDIM_X - 1 && threadIdx.y == 0)
		{
			temp_image[0][BLOCK_ELEMENT_X + 1] =
				get_element(
					original_extended_image,
					original_current_height - 1, 
					original_current_width + 4,
					original_width,
					original_pitch
				);
		}

		if (threadIdx.x == BLOCKDIM_X - 1 && threadIdx.y == BLOCKDIM_Y - 1)
		{
			temp_image[BLOCKDIM_Y + 1][BLOCK_ELEMENT_X + 1] = 
				get_element(
					original_extended_image,
					original_current_height + 1, 
					original_current_width + 4,
					original_width,
					original_pitch
				);
		}

		if (threadIdx.x == 0 && threadIdx.y == BLOCKDIM_Y - 1)
		{
			temp_image[BLOCKDIM_Y + 1][0] = 
				get_element(
					original_extended_image,
					original_current_height + 1, 
					original_current_width - 1,
					original_width,
					original_pitch
				);
		}
	}

	{
		if (threadIdx.x == 0)
		{
			temp_image[threadIdx.y + 1][0] = 
				get_element(
					original_extended_image,
					original_current_height,
				original_current_width - 1,
					original_width,
					original_pitch
				);
		}

		if (threadIdx.x == BLOCKDIM_X - 1)
		{
			temp_image[threadIdx.y + 1][BLOCK_ELEMENT_X + 1] = 
				get_element(
					original_extended_image,
					original_current_height,
				original_current_width + 4,
					original_width,
					original_pitch
				);
		}

		if (threadIdx.y == 0)
		{
			temp_image[0][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1] =
				get_element(
					original_extended_image,
					original_current_height - 1,
					original_current_width,
					original_width,
					original_pitch
				);

			temp_image[0][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2] =
				get_element(
					original_extended_image,
					original_current_height - 1,
					original_current_width + 1,
					original_width,
					original_pitch
				);

			temp_image[0][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3] =
				get_element(
					original_extended_image,
					original_current_height - 1,
					original_current_width + 2,
					original_width,
					original_pitch
				);

			temp_image[0][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 4] =
				get_element(
					original_extended_image,
					original_current_height - 1,
					original_current_width + 3,
					original_width,
					original_pitch
				);
		}

		if (threadIdx.y == BLOCKDIM_Y - 1)
		{
			temp_image[BLOCKDIM_Y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1] =
				get_element(
					original_extended_image,
					original_current_height + 1,
					original_current_width,
					original_width,
					original_pitch
				);

			temp_image[BLOCKDIM_Y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2] =
				get_element(
					original_extended_image,
					original_current_height + 1,
					original_current_width + 1,
					original_width,
					original_pitch
				);

			temp_image[BLOCKDIM_Y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 3] =
				get_element(
					original_extended_image,
					original_current_height + 1,
					original_current_width + 2,
					original_width,
					original_pitch
				);

			temp_image[BLOCKDIM_Y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 4] =
				get_element(
					original_extended_image,
					original_current_height + 1,
					original_current_width + 3,
					original_width,
					original_pitch
				);
		}
	}

	__syncthreads();

	image_result[get_index(result_current_height, result_current_width, result_width, result_pitch)] = (
		(
			temp_image[threadIdx.y		][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)    ] * (filter[0][0])
			+ temp_image[threadIdx.y	][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1] * (filter[0][1])
			+ temp_image[threadIdx.y	][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2] * (filter[0][2])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)    ] * (filter[1][0])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1] * (filter[1][1])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2] * (filter[1][2])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)    ] * (filter[2][0])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1] * (filter[2][1])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2] * (filter[2][2])
			)
		/ devision_coefficent
		);

	image_result[get_index(result_current_height, result_current_width + 1, result_width, result_pitch)] = (
		(
			temp_image[threadIdx.y		][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)     + 1] * (filter[0][0])
			+ temp_image[threadIdx.y	][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1] * (filter[0][1])
			+ temp_image[threadIdx.y	][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1] * (filter[0][2])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)     + 1] * (filter[1][0])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1] * (filter[1][1])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1] * (filter[1][2])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)     + 1] * (filter[2][0])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 1] * (filter[2][1])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 1] * (filter[2][2])
			)
		/ devision_coefficent
		);

	image_result[get_index(result_current_height, result_current_width + 2, result_width, result_pitch)] = (
		(
			temp_image[threadIdx.y		][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)     + 2] * (filter[0][0])
			+ temp_image[threadIdx.y	][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2] * (filter[0][1])
			+ temp_image[threadIdx.y	][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2] * (filter[0][2])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)     + 2] * (filter[1][0])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2] * (filter[1][1])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2] * (filter[1][2])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)     + 2] * (filter[2][0])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 2] * (filter[2][1])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 2] * (filter[2][2])
			)
		/ devision_coefficent
		);

	image_result[get_index(result_current_height, result_current_width + 3, result_width, result_pitch)] = (
		(
			temp_image[threadIdx.y		][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)     + 3] * (filter[0][0])
			+ temp_image[threadIdx.y	][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3] * (filter[0][1])
			+ temp_image[threadIdx.y	][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3] * (filter[0][2])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)     + 3] * (filter[1][0])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3] * (filter[1][1])
			+ temp_image[threadIdx.y + 1][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3] * (filter[1][2])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH)     + 3] * (filter[2][0])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 1 + 3] * (filter[2][1])
			+ temp_image[threadIdx.y + 2][(threadIdx.x * ELEMENT_IN_THREAD_WIDTH) + 2 + 3] * (filter[2][2])
			)
		/ devision_coefficent
		);
}

void check_cuda_status(cudaError_t cuda_status)
{
	if (cuda_status != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		exit(EXIT_FAILURE);
	}	
}

Result perform_GPU_shared_transactions_worker(Task task)
{
	cudaEvent_t start_time, stop_time;
	cudaEventCreate(&start_time);
	cudaEventCreate(&stop_time);

	size_t image_original_pitch;
	size_t image_result_pitch;

	unsigned char* image_original;
	unsigned char* image_result;

	auto cuda_status = cudaMallocPitch(
			(void**)(&image_original),
			&image_original_pitch,
			task.image.matrix.width * sizeof(unsigned char),
			task.image.matrix.height
		);
	check_cuda_status(cuda_status);

	cuda_status = cudaMemcpy2D(
		image_original,
		image_original_pitch,
		task.image.matrix.matrix,
		task.image.matrix.width * sizeof(unsigned char),
		task.image.matrix.width * sizeof(unsigned char),
		task.image.matrix.height,
		cudaMemcpyHostToDevice
	);
	check_cuda_status(cuda_status);

	cuda_status = cudaMallocPitch(
		(void**)(&image_result),
		&image_result_pitch,
		task.work_matrix.width * sizeof(unsigned char),
		task.image.matrix.height
	);
	check_cuda_status(cuda_status);

	dim3 block(BLOCKDIM_X, BLOCKDIM_Y);
	dim3 grid;

	grid.x = task.work_matrix.width / BLOCK_ELEMENT_X;
	if (task.work_matrix.width % BLOCK_ELEMENT_X != 0)
		grid.x += 1;

	grid.y = task.work_matrix.height / BLOCKDIM_Y;
	if (task.work_matrix.height % BLOCKDIM_Y != 0)
		grid.y += 1;

	cudaEventRecord(start_time);
	gpu_shared_transactions_filter<<<grid, block>>>(
		image_original,
		task.image.matrix.width,
		image_original_pitch,
		image_result,
		task.work_matrix.width,
		task.work_matrix.height,
		image_result_pitch,
		task.division_coef
	);
	cudaDeviceSynchronize();
	cudaEventRecord(stop_time);
	cudaEventSynchronize(stop_time);

	Result result;
	cudaEventElapsedTime(&result.time, start_time, stop_time);

	cuda_status = cudaMemcpy2D(
		task.work_matrix.matrix,
		task.work_matrix.width * sizeof(unsigned char),
		image_result,
		image_result_pitch,
		task.work_matrix.width * sizeof(unsigned char),
		task.work_matrix.height,
		cudaMemcpyDeviceToHost
	);
	check_cuda_status(cuda_status);

	result.result = task.work_matrix;
	cudaEventElapsedTime(&result.time, start_time, stop_time);
	return result;
}