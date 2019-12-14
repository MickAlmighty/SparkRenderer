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

	float NodeAI::measureManhattanDistance(glm::vec2 point) const
	{
		/*const float xDistance = glm::abs(position.x - point.x);
		const float yDistance = glm::abs(position.y - point.y);
		return xDistance + yDistance;*/

		//diagonal distance 
		const float D = 1.0f;
		const float D2 = 1.41f;
		const float xDistance = glm::abs(position.x - point.x);
		const float yDistance = glm::abs(position.y - point.y);
		return D * (xDistance + yDistance) + (D2 - 2 * D) * glm::min(xDistance, yDistance);

		//euclidean distance without sqrt
		//return glm::pow(position.x - endPoint.x, 2) + glm::pow(position.y - endPoint.y, 2);
		//return glm::distance(position, endPoint);
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
