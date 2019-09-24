#ifndef ACTOR_AI_H
#define ACTOR_AI_H
#include <Component.h>
#include <glm/glm.hpp>
#include <deque>

class Node
{
public:
	std::weak_ptr<Node> parent;
	glm::ivec2 position{};

	float distanceToEndPoint(glm::vec2 endPoint) const;
	std::list<std::shared_ptr<Node>> getNeighbors() const;
	void drawReturnPath(std::vector<glm::vec3>& perlinValues) const;
	void getPath(std::deque<std::pair<bool, glm::vec2>>& path) const;
	Node() = default;
	Node(glm::vec2 pos);
	~Node();
};

class ActorAI : public Component
{
	glm::ivec2 startPos{};
	glm::ivec2 endPos{};
	float timer = 0.0f;
	std::multimap<float, std::shared_ptr<Node>> nodesToProcess;
	std::list<std::shared_ptr<Node>> processedNodes;
	bool isTraveling = false;
	std::deque<std::pair<bool,glm::vec2>> path;
public:
	std::deque<std::pair<bool, glm::vec2>> findPath();
	std::shared_ptr<Node> getTheNearestNodeFromOpen();
	bool isNodeClosed(std::shared_ptr<Node> node);
	void walkToEndOfThePath();

	SerializableType getSerializableType() override;
	Json::Value serialize() override;
	void deserialize(Json::Value& root) override;
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;
	ActorAI(std::string&& newName = "ActorAI");
	~ActorAI();
};

#endif