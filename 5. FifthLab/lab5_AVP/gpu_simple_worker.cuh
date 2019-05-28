#pragma once
#include "task_creator.h"
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

Result perform_GPU_simple_worker(Task task);
