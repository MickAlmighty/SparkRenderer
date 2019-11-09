#ifndef MAP_CUH
#define MAP_CUH

#include <cuda_runtime.h>

namespace spark {
	namespace cuda {
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

			__device__ float getTerrainValue(const int x, const int y) const
			{
				const unsigned int index = getTerrainNodeIndex(x, y);
				return nodes[index];
			}

			__device__ int getTerrainNodeIndex(const int x, const int y) const
			{
				return x * width + y;
			}

			__device__ bool areIndexesValid(const int x, const int y) const
			{
				const bool validX = x >= 0 && x < width;
				const bool validY = y >= 0 && y < height;
				return validX && validY;
			}
		};

	}
}
#endif