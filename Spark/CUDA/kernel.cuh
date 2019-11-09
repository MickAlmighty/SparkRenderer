#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>

#include "Map.cuh"

namespace spark {
	namespace cuda {
		__global__ void checkMapValues(Map* mapDev);
		__global__ void attachNodes(Map* map, float* nodes);
		__global__ void findPath(Map* map, int* path);
		__host__ void runKernel(Map* mapDev, float* nodes, int* path);
	}
}

#endif