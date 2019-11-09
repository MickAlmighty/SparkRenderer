#ifndef NODE_CUH
#define NODE_CUH

#include <cuda_runtime.h>
#include <corecrt_math.h>

#include "Map.cuh"

namespace spark {
	namespace cuda {
		struct ivec2
		{
		public:
			int x{};
			int y{};
			__device__ ivec2() = default;
			__device__ ivec2(int x, int y)
			{
				this->x = x;
				this->y = y;
			}
			__device__ ivec2(int x)
			{
				this->x = x;
				this->y = x;
			}
		};


		class Node {
		public:
			Node* parent = nullptr;
			int pos[2] = {};
			float distanceFromBeginning{ 0 };
			float valueH{};
			float valueF{};

			__device__ Node() = default;

			__device__ Node(int* position, float distance) : distanceFromBeginning(distance)
			{
				pos[0] = position[0];
				pos[1] = position[1];
			}

			__device__ float measureManhattanDistance(int* point) const
			{
				const float xDistance = fabsf((float)(pos[0] - point[0]));
				const float yDistance = fabsf(static_cast<float>(pos[1] - point[1]));
				return xDistance + yDistance;
			}

			__device__ Node* getNeighbors(Map* map)
			{
				Node* nodes = new Node[8];
				const float distanceFromNode = 1.0f;
				const float diagonalDistanceFromNode = 1.41f;
				tryToCreateNeighbor(nodes + 0, { pos[0] - 1, pos[1] }, map, distanceFromNode);
				tryToCreateNeighbor(nodes + 1, { pos[0] + 1, pos[1] }, map, distanceFromNode);
				tryToCreateNeighbor(nodes + 2, { pos[0], pos[1] - 1 }, map, distanceFromNode);
				tryToCreateNeighbor(nodes + 3, { pos[0], pos[1] + 1 }, map, distanceFromNode);
				tryToCreateNeighbor(nodes + 4, { pos[0] - 1, pos[1] - 1 }, map, diagonalDistanceFromNode);
				tryToCreateNeighbor(nodes + 5, { pos[0] + 1, pos[1] - 1}, map, diagonalDistanceFromNode);
				tryToCreateNeighbor(nodes + 6, { pos[0] + 1, pos[1] + 1}, map, diagonalDistanceFromNode);
				tryToCreateNeighbor(nodes + 7, { pos[0] - 1, pos[1] + 1}, map, diagonalDistanceFromNode);
				return nodes;
			}

			__device__ void tryToCreateNeighbor(Node* child, ivec2 position,
				Map* map, const float depth) const
			{
				if (map->areIndexesValid(position.x, position.y))
				{
					if (map->getTerrainValue(pos[0], pos[1]) != 1.0f)
					{
						child->pos[0] = position.x;
						child->pos[1] = position.y;
						child->distanceFromBeginning = this->distanceFromBeginning + depth;
					}
				}
			}
		};
	}
}

#endif