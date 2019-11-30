#ifndef MAP_CUH
#define MAP_CUH

#include <cuda_runtime.h>

namespace spark {
	namespace cuda {
		class Map
		{
		public:
			float* nodes = nullptr;
			int width{};
			int height{};

			__host__ __device__ Map()
			{
				
			}

			__host__ __device__ Map(int width_, int height_, float* nodes_): width(width_), height(height_)
			{
				nodes = new float[width * height];
				memcpy(nodes, nodes_, width * height * sizeof(float));
			}

			Map& operator=(const Map& m)
			{
				if (this->nodes)
					delete[] nodes;

				this->width = m.width;
				this->height = m.height;

				nodes = new float[width * height];
				memcpy(nodes, m.nodes, width * height * sizeof(float));
				return *this;
			}

			__host__ __device__ ~Map()
			{
				delete[] nodes;
			}

			__host__ __device__ int getLength() const
			{
				return width * height;
			}

			__host__ __device__ float getTerrainValue(const int x, const int y) const
			{
				const unsigned int index = getTerrainNodeIndex(x, y);
				return nodes[index];
			}

			__host__ __device__ int getTerrainNodeIndex(const int x, const int y) const
			{
				return x * width + y;
			}

			__host__ __device__ bool areIndexesValid(const int x, const int y) const
			{
				const bool validX = x >= 0 && x < width;
				const bool validY = y >= 0 && y < height;
				return validX && validY;
			}
		};

	}
}
#endif