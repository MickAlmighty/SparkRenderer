#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>

namespace spark {
	namespace cuda {
		class Agent;
		class Map;
		__global__ void checkMapValues(Map* mapDev);
		__global__ void createMap(float* nodes, int width, int height);
		__global__ void findPath(int* path, int* memSize, Agent* agents);
		__global__ void fillPathBuffer(Agent* agents, int* pathBuffer, int numberOfAgents);
		__host__ void runKernel(int* path, int* memSize, Agent* agents);
		__host__ void initMap(float* nodes, int width, int height);
	}
}

#endif