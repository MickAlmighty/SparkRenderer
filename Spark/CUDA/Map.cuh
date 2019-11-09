#ifndef MAP_CUH
#define MAP_CUH

#include <cuda_runtime.h>

class Map
{
public:
	float* nodes = nullptr;
	int width;
	int height;

	__device__ int getLength() const
	{
		return width * height;
	}
	__device__ int getTerrainNodeIndex(const int x, const int y) const
	{
		return x * width + y;
	}
};

#endif