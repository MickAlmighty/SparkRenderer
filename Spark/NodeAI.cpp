#include "NodeAI.h"

#include "CUDA/Map.cuh"
#include "TerrainGenerator.h"

namespace spark {

	NodeAI::NodeAI(const glm::ivec2 pos, const float depth_) : position(pos), depth(depth_)
	{
	}

	NodeAI::NodeAI() : position(glm::ivec2(0, 0))
	{
	}

	float NodeAI::measureDistanceTo(glm::ivec2 point) const
	{
		/*const float xDistance = fabsf(static_cast<float>(pos[0] - point[0]));
		const float yDistance = fabsf(static_cast<float>(pos[1] - point[1]));
		return xDistance + yDistance;*/

		//diagonal non uniform cost
		const float D = 1.0f;
		const float D2 = 1.41f;
		const float xDistance = fabsf(static_cast<float>(position[0] - point[0]));
		const float yDistance = fabsf(static_cast<float>(position[1] - point[1]));
		return D * (xDistance + yDistance) + (D2 - 2 * D) * fmin(xDistance, yDistance);

		//diagonal uniform cost
		/*const float xDistance = fabsf(static_cast<float>(pos[0] - point[0]));
		const float yDistance = fabsf(static_cast<float>(pos[1] - point[1]));
		return fmax(xDistance, yDistance);*/
	}

	void NodeAI::calculateHeuristic(const cuda::Map& map, glm::ivec2 endPoint)
	{
		const float terrainValue = map.getTerrainValue(position[0], position[1]);
		functionF = (1.0f + terrainValue) * (measureDistanceTo(endPoint) + depth);
	}

	std::vector<NodeAI> NodeAI::getNeighbors(const cuda::Map& map) const
	{
		std::vector<NodeAI> neighbors;
		neighbors.reserve(8);
		const float distanceFromNode = 1.0f;
		const float diagonalDistanceFromNode = 1.41f;
		tryToCreateNeighbor(neighbors, { position.x - 1, position.y }, map, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x + 1, position.y }, map, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x, position.y - 1 }, map, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x, position.y + 1 }, map, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x - 1, position.y - 1 }, map, diagonalDistanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x + 1, position.y - 1 }, map, diagonalDistanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x + 1, position.y + 1 }, map, diagonalDistanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x - 1, position.y + 1 }, map, diagonalDistanceFromNode);

		return neighbors;
	}

	void NodeAI::getPath(std::deque<glm::ivec2>& path) const
	{
		path.push_front({ static_cast<float>(position.x), static_cast<float>(position.y) });
		if (parentAddress)
			parentAddress->getPath(path);
	}

	void NodeAI::tryToCreateNeighbor(std::vector<NodeAI>& container, glm::ivec2&& pos, const cuda::Map& map,
		const float depth) const
	{
		if (map.areIndexesValid(pos.x, pos.y))
		{
			if (map.getTerrainValue(pos.x, pos.y) != 1.0f)
			{
				if (parentAddress)
				{
					if (parentAddress->position == pos)
					{
						return;
					}
					container.emplace_back(pos, this->depth + depth);
				}
				else
				{
					container.emplace_back(pos, this->depth + depth);
				}
			}
		}
	}

}