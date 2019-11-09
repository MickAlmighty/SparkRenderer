#include "NodeAI.h"

#include "TerrainGenerator.h"
#include "Logging.h"

namespace spark {

NodeAI::NodeAI(const glm::ivec2 pos, const float depth_) : position(pos), depth(depth_)
{
    //SPARK_TRACE("NodeAI Constructor!");
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

float NodeAI::distanceToEndPoint(glm::vec2 endPoint) const
{
	return glm::pow(position.x - endPoint.x, 2) + glm::pow(position.y - endPoint.y, 2);
	//return glm::distance(position, endPoint);
}

std::list<std::shared_ptr<NodeAI>> NodeAI::getNeighbors(const std::shared_ptr<TerrainGenerator>& terrainGenerator) const
{
	std::list<std::shared_ptr<NodeAI>> neighbors;
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

std::list<NodeAI> NodeAI::getNeighborsStack(const std::shared_ptr<TerrainGenerator>& terrainGenerator) const
{
	std::list<NodeAI> neighbors;
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

void NodeAI::drawReturnPath(std::shared_ptr<TerrainGenerator>& terrainGenerator) const
{
	const int index = position.y * terrainGenerator->terrainSize + position.x;
	terrainGenerator->markNodeAsPartOfPath(position.x, position.y);
	if (!parent.expired())
		parent.lock()->drawReturnPath(terrainGenerator);
}

void NodeAI::getPath(std::deque<std::pair<bool, glm::ivec2>>& path) const
{
	path.push_front({ false, { static_cast<float>(position.x), static_cast<float>(position.y) } });
	if (!parent.expired())
		parent.lock()->getPath(path);
}

void NodeAI::drawReturnPathStack(std::shared_ptr<TerrainGenerator>& terrainGenerator) const
{
	const int index = position.y * terrainGenerator->terrainSize + position.x;
	terrainGenerator->markNodeAsPartOfPath(position.x, position.y);
	if (parentAddress)
		parentAddress->drawReturnPathStack(terrainGenerator);
}

void NodeAI::getPathStack(std::deque<std::pair<bool, glm::ivec2>>& path) const
{
	path.push_front({ false, { static_cast<float>(position.x), static_cast<float>(position.y) } });
	if (parentAddress)
		parentAddress->getPathStack(path);
}

	void NodeAI::tryToCreateNeighbor(std::list<std::shared_ptr<NodeAI>>& container, glm::ivec2&& pos,
	                                 const std::shared_ptr<TerrainGenerator>& terrainGenerator, const float depth) const
	{
		if (terrainGenerator->areIndicesValid(pos.x, pos.y))
		{
			if (terrainGenerator->getTerrainValue(pos.x, pos.y) != 1.0f)
			{
				container.emplace_back(std::make_shared<NodeAI>(pos, this->depth + depth));
			}
		}
	}

	void NodeAI::tryToCreateNeighbor(std::list<NodeAI>& container, glm::ivec2&& pos,
		const std::shared_ptr<TerrainGenerator>& terrainGenerator, const float depth) const
	{
		if (terrainGenerator->areIndicesValid(pos.x, pos.y))
		{
			if (terrainGenerator->getTerrainValue(pos.x, pos.y) != 1.0f)
			{
				container.emplace_back(pos, this->depth + depth);
			}
		}
	}

}
