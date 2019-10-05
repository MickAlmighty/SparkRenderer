#include "ActorAI.h"

#include <iostream>
#include <deque>

#include <glm/gtc/random.inl>

#include "Clock.h"
#include "GameObject.h"
#include "GUI/SparkGui.h"
#include "TerrainGenerator.h"
#include "JsonSerializer.h"

namespace spark {

float Node::distanceToEndPoint(glm::vec2 endPoint) const
{
	return glm::pow(position.x - endPoint.x, 2) + glm::pow(position.y - endPoint.y, 2);
	//return glm::distance(position, endPoint);
}

std::list<std::shared_ptr<Node>> Node::getNeighbors(const std::shared_ptr<TerrainGenerator>& terrainGenerator) const
{
	std::list<std::shared_ptr<Node>> neighbors;
	if (position.x - 1 >= 0)
	{
		if(const glm::ivec2 pos(position.x - 1, position.y); terrainGenerator->getTerrainValue(pos.x, pos.y) != 1.0f)
		{
			neighbors.push_back(std::make_shared<Node>(pos, this->depth + 1));
		}
	}
		
	if (position.x + 1 < 20) 
	{
		if (const glm::ivec2 pos(position.x + 1, position.y); terrainGenerator->getTerrainValue(pos.x, pos.y) != 1.0f)
		{
			neighbors.push_back(std::make_shared<Node>(pos, this->depth + 1));
		}
	}
	if (position.y - 1 >= 0)
	{
		if (const glm::ivec2 pos(position.x, position.y - 1); terrainGenerator->getTerrainValue(pos.x, pos.y) != 1.0f)
		{
			neighbors.push_back(std::make_shared<Node>(pos, this->depth + 1));
		}
	}
	if (position.y + 1 < 20)
	{
		if (const glm::ivec2 pos(position.x, position.y + 1); terrainGenerator->getTerrainValue(pos.x, pos.y) != 1.0f)
		{
			neighbors.push_back(std::make_shared<Node>(pos, this->depth + 1));
		}
	}
	/*if (position.x - 1 >= 0 && position.y - 1 >= 0)
		neighbors.push_back(std::make_shared<Node>(glm::ivec2(position.x - 1, position.y - 1)));
	if (position.x + 1 < 20 && position.y - 1 >= 0)
		neighbors.push_back(std::make_shared<Node>(glm::ivec2(position.x + 1, position.y - 1)));
	if (position.x + 1 < 20 && position.y + 1 < 20)
		neighbors.push_back(std::make_shared<Node>(glm::ivec2(position.x + 1, position.y + 1)));
	if (position.x - 1 >= 0 && position.y + 1 < 20)
		neighbors.push_back(std::make_shared<Node>(glm::ivec2(position.x - 1, position.y - 1)));*/
	return neighbors;
}

void Node::drawReturnPath(std::shared_ptr<TerrainGenerator>& terrainGenerator) const
{
	//std::cout << position.x << " " << position.y << std::endl;
	const int index = position.y * 20 + position.x;
	terrainGenerator->markNodeAsPartOfPath(position.x, position.y);
	if (!parent.expired())
		parent.lock()->drawReturnPath(terrainGenerator);
}

void Node::getPath(std::deque<std::pair<bool, glm::ivec2>>& path) const
{
	path.push_front({ false, { static_cast<float>(position.x), static_cast<float>(position.y) } });
	if (!parent.expired())
		parent.lock()->getPath(path);
}

Node::Node(const glm::ivec2 pos, const unsigned int depth_) : position(pos), depth(depth_)
{
	std::cout << "Node Constructor!" << std::endl;
}

Node::Node(const Node& rhs) : parent(rhs.parent), position(rhs.position), depth(rhs.depth)
{
	
}

Node::~Node()
{
	//std::cout << "Node destroyed" << std::endl;
}

void ActorAI::findPath()
{
	path.clear();
	nodesToProcess.emplace(0.0f, std::make_shared<Node>(startPos, 0));
	std::shared_ptr<Node> finishNode = nullptr;
	while (true)
	{
		if (nodesToProcess.empty())
		{
			std::cout << "Path hasn't been found" << std::endl;
			break;
		}

		const auto& closedNode = getTheNearestNodeFromOpen();
		if (closedNode->position == endPos)
		{
			finishNode = closedNode;
			break;
		}
		processedNodes.push_back(closedNode);
		std::list<std::shared_ptr<Node>> neighbors = closedNode->getNeighbors(terrainGenerator.lock());

		for (const std::shared_ptr<Node>& neighbor : neighbors)
		{
			if (isNodeClosed(neighbor))
				continue;

			neighbor->parent = closedNode;

			const float distance = neighbor->distanceToEndPoint(endPos);
			const float terrainValue = terrainGenerator.lock()->getTerrainValue(neighbor->position.x, neighbor->position.y);
			float heuristicsValue = terrainValue * (distance + neighbor->depth);
			
			nodesToProcess.insert({heuristicsValue, neighbor});
			//nodesToProcess.emplace(heuristicsValue, neighbor);
		}
	}
	if (finishNode)
	{
		if (auto terrain_g = terrainGenerator.lock(); terrain_g != nullptr)
		{
			finishNode->drawReturnPath(terrain_g);
			finishNode->getPath(path);
			terrain_g->updateTerrain();
		}
	}
	nodesToProcess.clear();
	processedNodes.clear();
}

std::shared_ptr<Node> ActorAI::getTheNearestNodeFromOpen()
{
	const auto node_it = std::begin(nodesToProcess);
	if (node_it != std::end(nodesToProcess))
	{
		std::shared_ptr<Node> n = node_it->second;
		nodesToProcess.erase(node_it);
		return n;
	}
	return nullptr;
}

bool ActorAI::isNodeClosed(const std::shared_ptr<Node>& node)
{
	const auto& it = std::find_if(std::begin(processedNodes), std::end(processedNodes), [&node](const std::shared_ptr<Node>& n)
	{
		return n->position == node->position;
	});
	return it != std::end(processedNodes);
}

void ActorAI::walkToEndOfThePath()
{
	if (path.empty())
		return;

	const auto wayPoint_it = std::find_if(path.begin(), path.end(),
		[](const std::pair<bool, glm::vec2>& p)
		{
			return !p.first;
		});
	if (wayPoint_it == path.end())
	{
		path.clear();
		isTraveling = false;
		return;
	}

	for (auto& wayPoint : path)
	{
		if (wayPoint.first)
		{
			continue;
		}

		if (!wayPoint.first)
		{
			const glm::vec3 position = getGameObject()->transform.world.getPosition();
			glm::vec3 pointOnPath = glm::vec3(wayPoint.second.x, 0.0f, wayPoint.second.y);
			if (glm::distance(position, pointOnPath) < 0.01f)
			{
				terrainGenerator.lock()->unMarkNodeAsPartOfPath(wayPoint.second.x, wayPoint.second.y);
				terrainGenerator.lock()->updateTerrain();
				wayPoint.first = true;
			}
			else
			{
				const glm::mat4 worldMatrix = getGameObject()->getParent()->transform.world.getMatrix();

				const glm::vec3 direction = glm::normalize(pointOnPath - position);
				const glm::vec3 updatedWorldPosition = position + direction * static_cast<float>(Clock::getDeltaTime()) * movementSpeed;
				const glm::vec4 localPosition = glm::inverse(worldMatrix) * glm::vec4(updatedWorldPosition, 1);
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
	root["movementSpeed"] = movementSpeed;
	root["terrainGenerator"] = JsonSerializer::serialize(terrainGenerator.lock());
	return root;
}

void ActorAI::deserialize(Json::Value& root)
{
	name = root.get("name", "ActorAI").asString();
	movementSpeed = root.get("movementSpeed", 1.0f).asFloat();
	terrainGenerator = std::dynamic_pointer_cast<TerrainGenerator>(JsonSerializer::deserialize(root["terrainGenerator"]));
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

	if (!isTraveling)
	{
		do
		{
			endPos = { static_cast<int>(glm::linearRand(0.0f, 20.0f)), static_cast<int>(glm::linearRand(0.0f, 20.0f)) };
		} while (terrainGenerator.lock()->getTerrainValue(endPos.x, endPos.y) == 1);
		
		const double measureStart = glfwGetTime();
		findPath();
		timer = glfwGetTime() - measureStart;
		isTraveling = true;
	}

	if (isTraveling)
	{
		walkToEndOfThePath();
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

	terrainGenerator = SparkGui::getObject("terrainGenerator", terrainGenerator.lock());

	ImGui::DragInt2("startPos", &startPos.x);
	ImGui::DragInt2("endPos", &endPos.x);
	ImGui::DragFloat("movementSpeed", &movementSpeed, 0.1f);
	ImGui::InputDouble("PathFindingTimeDuration", &timer, 0, 0, "%.8f");
	if (ImGui::Button("FindPath"))
	{
		const double measureStart = glfwGetTime();
		findPath();
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

}
