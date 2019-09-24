#include "ActorAI.h"
#include <GameObject.h>
#include <iostream>
#include "TerrainGenerator.h"
#include <deque>
#include "Clock.h"

float Node::distanceToEndPoint(glm::vec2 endPoint) const
{
	return glm::pow(position.x - endPoint.x, 2) + glm::pow(position.y - endPoint.y, 2);
	//return glm::distance(position, endPoint);
}

std::list<std::shared_ptr<Node>> Node::getNeighbors() const
{
	std::list<std::shared_ptr<Node>> neighbors;
	if (position.x - 1 >= 0)
		neighbors.push_back(std::make_shared<Node>(glm::vec2(position.x - 1, position.y)));
	if (position.x + 1 < 20)
		neighbors.push_back(std::make_shared<Node>(glm::vec2(position.x + 1, position.y)));
	if (position.y - 1 >= 0)
		neighbors.push_back(std::make_shared<Node>(glm::vec2(position.x, position.y - 1)));
	if (position.y + 1 < 20)
		neighbors.push_back(std::make_shared<Node>(glm::vec2(position.x, position.y + 1)));
	if (position.x - 1 >= 0 && position.y - 1 >= 0)
		neighbors.push_back(std::make_shared<Node>(glm::vec2(position.x - 1, position.y - 1)));
	if (position.x + 1 < 20 && position.y - 1 >= 0)
		neighbors.push_back(std::make_shared<Node>(glm::vec2(position.x + 1, position.y - 1)));
	if (position.x + 1 < 20 && position.y + 1 < 20)
		neighbors.push_back(std::make_shared<Node>(glm::vec2(position.x + 1, position.y + 1)));
	if (position.x - 1 >= 0 && position.y + 1 < 20)
		neighbors.push_back(std::make_shared<Node>(glm::vec2(position.x - 1, position.y - 1)));
	return neighbors;
}

void Node::drawReturnPath(std::vector<glm::vec3>& perlinValues) const
{
	//std::cout << position.x << " " << position.y << std::endl;
	int index = position.y * 20 + position.x;
	perlinValues[index].y = 1.0f;
	if (!parent.expired())
		parent.lock()->drawReturnPath(perlinValues);
}

void Node::getPath(std::deque<glm::vec2>& path) const
{
	path.push_front({ static_cast<float>(position.x), static_cast<float>(position.y) });
	if (!parent.expired())
		parent.lock()->getPath(path);
}

Node::Node(glm::vec2 pos)
{
	position = pos;
}

Node::~Node()
{
	//std::cout << "Node destroyed" << std::endl;
}

std::deque<glm::vec2> ActorAI::findPath()
{
	bool isPathFound = false;
	glm::vec2 finish = endPos;
	
	nodesToProcess.emplace(0, new Node(startPos));
	std::shared_ptr<Node> finishNode = nullptr;
	while (!isPathFound)
	{
		if (nodesToProcess.empty())
		{
			std::cout << "Path hasn't been found" << std::endl;
			break;
		}

		const auto& closedNode = getTheNearestNodeFromOpen();
		if(closedNode->position == endPos)
		{
			finishNode = closedNode;
			isPathFound = true;
			break;
		}
		processedNodes.push_back(closedNode);
		std::list<std::shared_ptr<Node>> neighbors = closedNode->getNeighbors();

		for (std::shared_ptr<Node> neighbor : neighbors)
		{
			if (isNodeClosed(neighbor))
				continue;

			neighbor->parent = closedNode;
			float distance = neighbor->distanceToEndPoint(endPos);
			nodesToProcess.emplace(distance, neighbor);

		}
	}
	std::deque<glm::vec2> path;
	if(finishNode)
	{
		auto perlinValues = getGameObject()->getComponent<TerrainGenerator>()->getPerlinValues();
		finishNode->drawReturnPath(perlinValues);
		finishNode->getPath(path);
		getGameObject()->getComponent<TerrainGenerator>()->updateTerrain(perlinValues);
	}
	nodesToProcess.clear();
	processedNodes.clear();
	return path;
}

std::shared_ptr<Node> ActorAI::getTheNearestNodeFromOpen()
{
	const auto node_it = std::begin(nodesToProcess);
	if(node_it != std::end(nodesToProcess))
	{
		std::shared_ptr<Node> n = node_it->second;
		nodesToProcess.erase(node_it);
		return n;
	}
	return nullptr;
}

bool ActorAI::isNodeClosed(std::shared_ptr<Node> node)
{
	const auto& it = std::find_if(std::begin(processedNodes), std::end(processedNodes), [&node](const std::shared_ptr<Node>& n)
	{
		return n->position == node->position;
	});
	return it != std::end(processedNodes);
}

void ActorAI::walkToEndOfThePath(std::deque<glm::vec2>& path)
{
	float deltaTime = Clock::getDeltaTime();
	glm::vec3 position = getGameObject()->transform.world.getPosition();
	glm::mat4 worldMatrix = getGameObject()->getParent()->transform.world.getMatrix();
	if (path.empty() && nodesPassed.empty())
		return;
	
	if(std::find(nodesPassed.begin(), nodesPassed.end(), false) == nodesPassed.end())
	{
		path.clear();
		nodesPassed.clear();
		isTraveling = false;
		return;
	}

	for(int i = 0; i < path.size(); i++)
	{
		if(nodesPassed[i])
		{
			continue;
		}

		if(!nodesPassed[i])
		{
			glm::vec3 waypoint = glm::vec3(path[i].x, 0.0f, path[i].y);
			if(glm::distance(position, waypoint) < 0.1f)
			{
				nodesPassed[i] = true;
			}
			else
			{
				glm::vec3 direction = glm::normalize(waypoint - position);
				glm::vec3 updatedWorldPosition = position + direction * deltaTime;
				glm::vec4 localPosition = glm::inverse(worldMatrix) * glm::vec4(updatedWorldPosition, 1);
				getGameObject()->transform.local.setPosition(localPosition);
				break;
			}
		}
	}
}

SerializableType ActorAI::getSerializableType()
{
	return SerializableType::SActorAI;
}

Json::Value ActorAI::serialize()
{
	Json::Value root;
	root["name"] = name;
	return root;
}

void ActorAI::deserialize(Json::Value& root)
{
	name = root.get("name", "ActorAI").asString();
}

void ActorAI::update()
{
	glm::vec3 pos = getGameObject()->transform.world.getPosition();
	if (pos.x < 0)
		pos.x = 0;
	if (pos.z < 0)
		pos.z = 0;
	if (pos.x > 19)
		pos.x = 19;
	if (pos.z > 19)
		pos.z = 19;
	if (pos.x - static_cast<int>(pos.x) < 0.5)
	{
		startPos.x = static_cast<int>(pos.x);
	}
	else
	{
		startPos.x = static_cast<int>(pos.x + 1);
	}

	if (pos.z - static_cast<int>(pos.z) < 0.5)
	{
		startPos.y = static_cast<int>(pos.z);
	}
	else
	{
		startPos.y = static_cast<int>(pos.z + 1);
	}
	
	if(isTraveling)
	{
		walkToEndOfThePath(path);
	}
	
}

void ActorAI::fixedUpdate()
{

}

void ActorAI::drawGUI()
{
	ImGui::PushID(this);
	ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
	ImGui::SetNextWindowSizeConstraints(ImVec2(250, 0), ImVec2(FLT_MAX, 150)); // Width = 250, Height > 100
	ImGui::BeginChild("ActorAI", { 0, 0 }, true, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_AlwaysAutoResize);
	if (ImGui::BeginMenuBar())
	{
		ImGui::Text("ActorAI");
		ImGui::EndMenuBar();
	}

	ImGui::DragInt2("startPos", &startPos.x);
	ImGui::DragInt2("endPos", &endPos.x);
	ImGui::InputFloat("PathFindingTimeDuration", &timer, 0, 0, "%.8f");
	if(ImGui::Button("FindPath"))
	{
		const float measureStart = glfwGetTime();
		path = findPath();
		nodesPassed = std::vector<bool>(path.size());
		timer = glfwGetTime() - measureStart;
		isTraveling = true;
	}
	removeComponentGUI<ActorAI>();

	ImGui::EndChild();
	ImGui::PopStyleVar();
	ImGui::PopID();
}

ActorAI::ActorAI(std::string&& newName) : Component(newName)
{
}


ActorAI::~ActorAI()
{
}
