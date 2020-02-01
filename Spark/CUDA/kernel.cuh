#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda.h>
#include <cuda_runtime.h>

namespace spark {
	enum class PathFindingMode : unsigned char;
}

namespace spark {
	namespace cuda {
		class Node;
		class Map;

		__global__ void checkMapValues(Map* mapDev);
		__global__ void createMap(float* nodes, int width, int height);
		__global__ void findPath(int* path, unsigned int* agentPaths, void* kernelMemory, const int maxThreadsPerBlock, bool showPathInfo = false);
		__host__ void runKernel(int blocks, int threads, int* path, unsigned int* agentPaths, void* kernelMemory, const Map& mapHost, PathFindingMode mode, const int maxThreadsPerBlock);
		__host__ void initMap(float* nodes, int width, int height);
		inline __device__ void constructPath(unsigned int* path, Node* const finishNode, Node* closedNodes, bool showPathInfo);
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