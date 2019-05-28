#pragma once
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "task_creator.h"

Result perform_GPU_shared_transactions_worker(Task task);