#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include "Map.cuh"

__global__ void checkMapValues(Map* mapDev);
__global__ void attachNodes(Map* map, float* nodes);
__host__ void runKernel(Map* mapDev, float* nodes);