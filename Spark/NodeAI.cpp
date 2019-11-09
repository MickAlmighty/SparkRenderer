#include "NodeAI.h"

#include "TerrainGenerator.h"

namespace spark {

	NodeAI::NodeAI(const glm::ivec2 pos, const float depth_) : position(pos), depth(depth_)
	{
		//std::cout << "NodeAI Constructor!" << std::endl;
	}

	NodeAI::NodeAI(const NodeAI& rhs) : position(rhs.position), depth(rhs.depth), parentAddress(rhs.parentAddress)
	{
		if (!rhs.parent.expired())
		{
			parent = rhs.parent;
		}
	}

	NodeAI::NodeAI(const NodeAI&& rhs) noexcept : position(rhs.position), depth(rhs.depth), parentAddress(rhs.parentAddress)
	{
	}

	NodeAI::NodeAI() : position(glm::ivec2(0, 0))
	{
	}

	float NodeAI::measureManhattanDistance(glm::vec2 point) const
	{
		const float xDistance = glm::abs(position.x - point.x);
		const float yDistance = glm::abs(position.y - point.y);
		return xDistance + yDistance;
		//return glm::pow(position.x - endPoint.x, 2) + glm::pow(position.y - endPoint.y, 2);
		//return glm::distance(position, endPoint);
	}

	std::vector<std::shared_ptr<NodeAI>> NodeAI::getNeighbors(const std::shared_ptr<TerrainGenerator>& terrainGenerator) const
	{
		std::vector<std::shared_ptr<NodeAI>> neighbors;
		neighbors.reserve(8);
		const float distanceFromNode = 1.0f;
		const float diagonalDistanceFromNode = 1.41f;
		tryToCreateNeighbor(neighbors, { position.x - 1, position.y }, terrainGenerator, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x + 1, position.y }, terrainGenerator, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x, position.y - 1 }, terrainGenerator, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x, position.y + 1 }, terrainGenerator, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x - 1, position.y - 1 }, terrainGenerator, diagonalDistanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x + 1, position.y - 1 }, terrainGenerator, diagonalDistanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x + 1, position.y + 1 }, terrainGenerator, diagonalDistanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x - 1, position.y + 1 }, terrainGenerator, diagonalDistanceFromNode);

		return neighbors;
	}

	std::vector<NodeAI> NodeAI::getNeighborsStack(const std::shared_ptr<TerrainGenerator>& terrainGenerator) const
	{
		std::vector<NodeAI> neighbors;
		neighbors.reserve(8);
		const float distanceFromNode = 1.0f;
		const float diagonalDistanceFromNode = 1.41f;
		tryToCreateNeighbor(neighbors, { position.x - 1, position.y }, terrainGenerator, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x + 1, position.y }, terrainGenerator, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x, position.y - 1 }, terrainGenerator, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x, position.y + 1 }, terrainGenerator, distanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x - 1, position.y - 1 }, terrainGenerator, diagonalDistanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x + 1, position.y - 1 }, terrainGenerator, diagonalDistanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x + 1, position.y + 1 }, terrainGenerator, diagonalDistanceFromNode);
		tryToCreateNeighbor(neighbors, { position.x - 1, position.y + 1 }, terrainGenerator, diagonalDistanceFromNode);

		return neighbors;
	}

	void NodeAI::getPath(std::deque<std::pair<bool, glm::ivec2>>& path) const
	{
		path.push_front({ false, { static_cast<float>(position.x), static_cast<float>(position.y) } });
		if (!parent.expired())
			parent.lock()->getPath(path);
	}

	void NodeAI::getPathStack(std::deque<std::pair<bool, glm::ivec2>>& path) const
	{
		path.push_front({ false, { static_cast<float>(position.x), static_cast<float>(position.y) } });
		if (parentAddress)
			parentAddress->getPathStack(path);
	}

	void NodeAI::tryToCreateNeighbor(std::vector<std::shared_ptr<NodeAI>>& container, glm::ivec2&& pos,
	                                 const std::shared_ptr<TerrainGenerator>& terrainGenerator, const float depth) const
	{
		if (terrainGenerator->areIndexesValid(pos.x, pos.y))
		{
			if (terrainGenerator->getTerrainValue(pos.x, pos.y) != 1.0f)
			{
				container.emplace_back(std::make_shared<NodeAI>(pos, this->depth + depth));
			}
		}
	}

	void NodeAI::tryToCreateNeighbor(std::vector<NodeAI>& container, glm::ivec2&& pos,
		const std::shared_ptr<TerrainGenerator>& terrainGenerator, const float depth) const
	{
		if (terrainGenerator->areIndexesValid(pos.x, pos.y))
		{
			if (terrainGenerator->getTerrainValue(pos.x, pos.y) != 1.0f)
			{
				container.emplace_back(pos, this->depth + depth);
			}
		}
	}

}
