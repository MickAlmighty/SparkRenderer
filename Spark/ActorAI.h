#ifndef ACTOR_AI_H
#define ACTOR_AI_H

#include <deque>

#include <glm/glm.hpp>

#include <Component.h>

namespace spark {

class Node
{
public:
	std::weak_ptr<Node> parent;
	glm::ivec2 position{};

	Node() = default;
	Node(glm::vec2 pos);
	~Node();

	float distanceToEndPoint(glm::vec2 endPoint) const;
	std::list<std::shared_ptr<Node>> getNeighbors() const;
	void drawReturnPath(std::vector<glm::vec3>& perlinValues) const;
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

	void findPath();
	std::shared_ptr<Node> getTheNearestNodeFromOpen();
	bool isNodeClosed(std::shared_ptr<Node> node);
	void walkToEndOfThePath();
};
}
#endif