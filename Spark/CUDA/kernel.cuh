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
		__global__ void attachNodes(Map* map, float* nodes);
		__global__ void findPath(Map* map, int* path, int* memSize, Agent* agents);
		__global__ void fillPathBuffer(Agent* agents, int* pathBuffer, int numberOfAgents);
		__host__ void runKernel(Map* mapDev, float* nodes, int* path, int* memSize, Agent* agents);
	}
	}

#endif