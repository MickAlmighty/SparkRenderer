#ifndef NODE_CUH
#define NODE_CUH

#include "Map.cuh"
#include "List.cuh"

namespace spark {
	namespace cuda {
		struct ivec2
		{
		public:
			int x{};
			int y{};
			__device__ ivec2() {}
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
			float valueF{};
			int pos[2] = {};
			float distanceFromBeginning{ 0 };
			float valueH{};
			Node* parent = nullptr;

			__device__ Node() {}

			__device__ Node(int* position, float distance) : distanceFromBeginning(distance)
			{
				pos[0] = position[0];
				pos[1] = position[1];
			}

			__device__ bool operator<(const Node& node) const
			{
				return this->valueF < node.valueF;
			}

			__device__ bool operator==(const Node& node) const
			{
				const bool xpos = this->pos[0] == node.pos[0];
				const bool ypos = this->pos[1] == node.pos[1];
				return xpos && ypos;
			}

			__device__ float measureManhattanDistance(int* point) const
			{
				const float xDistance = fabsf(static_cast<float>(pos[0] - point[0]));
				const float yDistance = fabsf(static_cast<float>(pos[1] - point[1]));
				return xDistance + yDistance;
			}

			__device__ List<Node> getNeighbors(Map* map)
			{
				List<Node> nodes;
				const float distanceFromNode = 1.0f;
				const float diagonalDistanceFromNode = 1.41f;
				tryToCreateNeighbor(nodes, { pos[0] - 1, pos[1] }, map, distanceFromNode);
				tryToCreateNeighbor(nodes, { pos[0] + 1, pos[1] }, map, distanceFromNode);
				tryToCreateNeighbor(nodes, { pos[0], pos[1] - 1 }, map, distanceFromNode);
				tryToCreateNeighbor(nodes, { pos[0], pos[1] + 1 }, map, distanceFromNode);
				tryToCreateNeighbor(nodes, { pos[0] - 1, pos[1] - 1 }, map, diagonalDistanceFromNode);
				tryToCreateNeighbor(nodes, { pos[0] + 1, pos[1] - 1}, map, diagonalDistanceFromNode);
				tryToCreateNeighbor(nodes, { pos[0] + 1, pos[1] + 1}, map, diagonalDistanceFromNode);
				tryToCreateNeighbor(nodes, { pos[0] - 1, pos[1] + 1}, map, diagonalDistanceFromNode);
				return nodes;
			}

			__device__ void tryToCreateNeighbor(List<Node>& list, ivec2 position,
				Map* map, const float depth) const
			{
				if (map->areIndexesValid(position.x, position.y))
				{
					if (map->getTerrainValue(position.x, position.y) != 1.0f)
					{
						Node child;
						child.pos[0] = position.x;
						child.pos[1] = position.y;
						child.distanceFromBeginning = this->distanceFromBeginning + depth;
						list.insert(child);
					}
				}
			}

			__device__ void getPathLength(int& length) const
			{
				++length;
				if (parent != nullptr)
				{
					parent->getPathLength(length);
				}
			}
			
			__device__ void recreatePath(int* path, int index)
			{
				index = index - 1;
				path[index * 2] = pos[0];
				path[index * 2 + 1] = pos[1];
				if (parent != nullptr)
				{
					parent->recreatePath(path, index);
				}
			}
		};
	}
}

#endif