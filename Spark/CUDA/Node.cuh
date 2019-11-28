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
			bool valid = false;
			Node* parent = nullptr;

			__device__ Node() {}

			__device__ Node(int* position, float distance) : distanceFromBeginning(distance)
			{
				pos[0] = position[0];
				pos[1] = position[1];
				valid = true;
			}

			__device__ Node(const Node& n) 
				: valueF(n.valueF), 
				distanceFromBeginning(n.distanceFromBeginning), 
				valueH(n.valueH), 
				valid(n.valid), 
				parent(n.parent)
			{
				pos[0] = n.pos[0];
				pos[1] = n.pos[1];
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

			__device__ void getNeighbors(Map* map, Node* nodes)
			{
				const float distanceFromNode = 1.0f;
				const float diagonalDistanceFromNode = 1.41f;
				tryToCreateNeighbor(nodes, { pos[0] - 1, pos[1] }, map, distanceFromNode);
				tryToCreateNeighbor(nodes + 1, { pos[0] + 1, pos[1] }, map, distanceFromNode);
				tryToCreateNeighbor(nodes + 2, { pos[0], pos[1] - 1 }, map, distanceFromNode);
				tryToCreateNeighbor(nodes + 3, { pos[0], pos[1] + 1 }, map, distanceFromNode);
				tryToCreateNeighbor(nodes + 4, { pos[0] - 1, pos[1] - 1 }, map, diagonalDistanceFromNode);
				tryToCreateNeighbor(nodes + 5, { pos[0] + 1, pos[1] - 1}, map, diagonalDistanceFromNode);
				tryToCreateNeighbor(nodes + 6, { pos[0] + 1, pos[1] + 1}, map, diagonalDistanceFromNode);
				tryToCreateNeighbor(nodes + 7, { pos[0] - 1, pos[1] + 1}, map, diagonalDistanceFromNode);
			}

			__device__ void tryToCreateNeighbor(Node* node, ivec2 position,
				Map* map, const float depth) const
			{
				if (map->areIndexesValid(position.x, position.y))
				{
					if (map->getTerrainValue(position.x, position.y) != 1.0f)
					{
						node->pos[0] = position.x;
						node->pos[1] = position.y;
						node->distanceFromBeginning = this->distanceFromBeginning + depth;
						node->valid = true;
					}
					else
					{
						node->valid = false;
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
			
			__device__ void recreatePath(unsigned int* path, int index)
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