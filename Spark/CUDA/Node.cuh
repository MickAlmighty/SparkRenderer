#ifndef NODE_CUH
#define NODE_CUH

#include <deque>

#include "Map.cuh"

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
			uint16_t pos[2] = {};
			float valueF{};
			float distanceFromBeginning{ 0 };
			uint16_t valid{ 0 };
			uint16_t parentIdx{ 65'535 };

			__device__ Node() {}

			__device__ Node(const int* position, float distance) : distanceFromBeginning(distance)
			{
				pos[0] = position[0];
				pos[1] = position[1];
				valid = 1;
			}

			__device__ Node(const Node& n)
				:
				valueF(n.valueF),
				distanceFromBeginning(n.distanceFromBeginning),
				valid(n.valid),
				parentIdx(n.parentIdx)
			{
				pos[0] = n.pos[0];
				pos[1] = n.pos[1];
			}

			__device__ bool operator<(const Node& node) const
			{
				if (valueF == node.valueF)
				{
					return distanceFromBeginning > node.distanceFromBeginning;
				}

				return valueF < node.valueF;
			}

			__device__ bool operator==(const Node& node) const
			{
				//const bool xpos = this->pos[0] == node.pos[0];
				//const bool ypos = this->pos[1] == node.pos[1];
				//return xpos && ypos;
				return !(*this < node) && !(node < *this);
			}

			__device__ float measureDistanceTo(int* point) const
			{
				/*const float xDistance = fabsf(static_cast<float>(pos[0] - point[0]));
				const float yDistance = fabsf(static_cast<float>(pos[1] - point[1]));
				return xDistance + yDistance;*/

				//diagonal non uniform cost
				const float D = 1.0f;
				const float D2 = 1.41f;
				const float xDistance = fabsf(static_cast<float>(pos[0] - point[0]));
				const float yDistance = fabsf(static_cast<float>(pos[1] - point[1]));
				return D * (xDistance + yDistance) + (D2 - 2 * D) * fmin(xDistance, yDistance);

				//diagonal uniform cost
				/*const float xDistance = fabsf(static_cast<float>(pos[0] - point[0]));
				const float yDistance = fabsf(static_cast<float>(pos[1] - point[1]));
				return fmax(xDistance, yDistance);*/
			}

			__device__ void calculateHeuristic(const Map* map, int* endPoint)
			{
				const float terrainValue = map->getTerrainValue(pos[0], pos[1]);
				valueF = (1.0f + terrainValue) * (measureDistanceTo(endPoint) + distanceFromBeginning);
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
				tryToCreateNeighbor(nodes + 5, { pos[0] + 1, pos[1] - 1 }, map, diagonalDistanceFromNode);
				tryToCreateNeighbor(nodes + 6, { pos[0] + 1, pos[1] + 1 }, map, diagonalDistanceFromNode);
				tryToCreateNeighbor(nodes + 7, { pos[0] - 1, pos[1] + 1 }, map, diagonalDistanceFromNode);
			}

			__device__ void tryToCreateNeighbor(Node* node, ivec2 position,
				Map* map, const float depth) const
			{
				if (!map->areIndexesValid(position.x, position.y) ||
					map->getTerrainValue(position.x, position.y) == 1.0f)
				{
					node->valid = 0;
					return;
				}

				node->pos[0] = position.x;
				node->pos[1] = position.y;
				node->distanceFromBeginning = this->distanceFromBeginning + depth;
				node->valid = 1;
			}
		};
	}
}

#endif