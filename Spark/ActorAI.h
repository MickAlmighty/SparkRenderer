#ifndef ACTOR_AI_H
#define ACTOR_AI_H

#include <deque>

#include <glm/glm.hpp>

#include <Component.h>

namespace spark {
	
class TerrainGenerator;

class Node
{
public:
	std::weak_ptr<Node> parent;
	const glm::ivec2 position;
	unsigned int depth = 0;

	Node(const glm::ivec2 pos, const unsigned int depth_);
	Node(const Node& rhs);
	~Node();

	float distanceToEndPoint(glm::vec2 endPoint) const;
	std::list<std::shared_ptr<Node>> getNeighbors(const std::shared_ptr<TerrainGenerator>& terrainGenerator) const;
	void drawReturnPath(std::shared_ptr<TerrainGenerator>& terrainGenerator) const;
	void getPath(std::deque<std::pair<bool, glm::ivec2>>& path) const;
};

class ActorAI : public Component
{
public:
	ActorAI(std::string&& newName = "ActorAI");
	~ActorAI();

	SerializableType getSerializableType() override;
	Json::Value serialize() override;
	void deserialize(Json::Value& root) override;
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;

private:
	double timer = 0.0f;
	float movementSpeed = 1.0f;
	bool isTraveling = false;
	glm::ivec2 startPos{};
	glm::ivec2 endPos{};
	std::multimap<float, std::shared_ptr<Node>> nodesToProcess;
	std::list<std::shared_ptr<Node>> processedNodes;
	std::deque<std::pair<bool, glm::ivec2>> path;
	std::weak_ptr<TerrainGenerator> terrainGenerator;

	void findPath();
	std::shared_ptr<Node> getTheNearestNodeFromOpen();
	bool isNodeClosed(const std::shared_ptr<Node>& node);
	void walkToEndOfThePath();
};
}
#endif