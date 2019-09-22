#ifndef ACTOR_AI_H
#define ACTOR_AI_H
#include <Component.h>
#include <glm/glm.hpp>

class Node
{
public:
	Node() = default;
	Node(glm::vec2 pos)
	{
		position = pos;
	};
	~Node() = default;
	Node* parent = nullptr;
	glm::vec2 position{};

	float distanceToEndPoint(glm::vec2 endPoint) const
	{
		return glm::distance(position, endPoint);
	}
	std::list<Node*> getNeighbors() const
	{
		std::list<Node*> neighbors;
		if (position.x - 1 >= 0)
			neighbors.push_back(new Node(glm::vec2(position.x - 1, position.y)));
		if (position.x + 1 <= 20)
			neighbors.push_back(new Node(glm::vec2(position.x + 1, position.y)));
		if (position.y - 1 >= 0)
			neighbors.push_back(new Node(glm::vec2(position.x, position.y - 1)));
		if (position.y + 1 <= 20)
			neighbors.push_back(new Node(glm::vec2(position.x, position.y + 1)));
		return neighbors;
	}
	void drawReturnPath(std::vector<glm::vec3>& perlinValues) const;
};

class ActorAI : public Component
{
	glm::ivec2 startPos{};
	glm::ivec2 endPos{};
	std::multimap<float, Node*> nodesToProcess;
	std::list<Node*> processedNodes;
	Node map[20][20] = {};
public:
	void findPath();
	void createMap();
	Node* getTheNearestNodeFromOpen();
	bool isNodeClosed(Node* node);
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