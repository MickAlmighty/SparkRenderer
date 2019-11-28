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
		__global__ void findPath(int* path, unsigned int* agentPaths);
		__host__ void runKernel(int* path, unsigned int* agentPaths);
		__host__ void initMap(float* nodes, int width, int height);
	}
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert, error %d: %s %s %d\n", code, cudaGetErrorString(code), file, line);
		/*if (abort) 
			exit(code);*/
	}
}

#endif