#include "ActorAI.h"

#include <deque>
#include <iostream>
#include <math.h>

#include <glm/gtc/random.inl>

#include "Clock.h"
#include "GameObject.h"
#include "GUI/SparkGui.h"
#include "JsonSerializer.h"
#include "NodeAI.h"
#include "TerrainGenerator.h"


namespace spark {

void ActorAI::update()
{
	if(terrainGenerator.expired())
		return;

	glm::vec3 pos = getGameObject()->transform.world.getPosition();
	
	validateActorPosition(pos);
	setStartPosition(pos);

	if (!isTraveling)
	{
		do
		{
			endPos = { static_cast<int>(glm::linearRand(0.0f, 20.0f)), static_cast<int>(glm::linearRand(0.0f, 20.0f)) };
		} while (terrainGenerator.lock()->getTerrainValue(endPos.x, endPos.y) == 1.0f);

		const double measureStart = glfwGetTime();
		//findPath();
		findPathStack();
		timer = glfwGetTime() - measureStart;
		std::cout << timer * 1000.0 << " ms" << std::endl;
		if (!path.empty())
		{
			isTraveling = true;
		}
	}

	if (isTraveling)
	{
		walkToEndOfThePath();
	}
}

void ActorAI::validateActorPosition(glm::vec3& position) const
{
	if (position.x < 0) position.x = 0;
	if (position.z < 0) position.z = 0;
	if (position.x > 19) position.x = 19;
	if (position.z > 19) position.z = 19;
}

void ActorAI::setStartPosition(glm::vec3& position)
{
	if (position.x - static_cast<int>(position.x) < 0.5)
	{
		startPos.x = static_cast<int>(position.x);
	}
	else
	{
		startPos.x = static_cast<int>(position.x + 1);
	}

	if (position.z - static_cast<int>(position.z) < 0.5)
	{
		startPos.y = static_cast<int>(position.z);
	}
	else
	{
		startPos.y = static_cast<int>(position.z + 1);
	}
}

void ActorAI::fixedUpdate()
{

}

void ActorAI::findPath()
{
	path.clear();
	nodesToProcess.emplace(0.0f, std::make_shared<NodeAI>(startPos, 0.0f));
	std::shared_ptr<NodeAI> finishNode = nullptr;
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
		std::list<std::shared_ptr<NodeAI>> neighbors = closedNode->getNeighbors(terrainGenerator.lock());

		for (const std::shared_ptr<NodeAI>& neighbor : neighbors)
		{
			if (isNodeClosed(neighbor))
				continue;

			neighbor->parent = closedNode;

			const float distance = neighbor->distanceToEndPoint(endPos);
			const float terrainValue = terrainGenerator.lock()->getTerrainValue(neighbor->position.x, neighbor->position.y);
			float heuristicsValue = (1 - terrainValue) * (distance + neighbor->depth);
			
			nodesToProcess.emplace(heuristicsValue, neighbor);
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

std::shared_ptr<NodeAI> ActorAI::getTheNearestNodeFromOpen()
{
	const auto node_it = std::begin(nodesToProcess);
	if (node_it != std::end(nodesToProcess))
	{
		std::shared_ptr<NodeAI> n = node_it->second;
		nodesToProcess.erase(node_it);
		return n;
	}
	return nullptr;
}

bool ActorAI::isNodeClosed(const std::shared_ptr<NodeAI>& node)
{
	const auto& it = std::find_if(std::begin(processedNodes), std::end(processedNodes), [&node](const std::shared_ptr<NodeAI>& n)
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

void ActorAI::findPathStack()
{
	path.clear();
	nodesToProcessStack.emplace( 0.0f, NodeAI(startPos, 0.0f));
	NodeAI* finishNode = nullptr;
	while (true)
	{
		if (nodesToProcessStack.empty())
		{
			isTraveling = false;
			std::cout << "Path hasn't been found" << std::endl;
			break;
		}

		const auto closedNode = getTheNearestNodeFromOpenStack();
		processedNodesStack.push_back(closedNode);
		
		if (closedNode.position == endPos)
		{
			finishNode = &(*std::prev(processedNodesStack.end()));
			break;
		}
		
		std::list<NodeAI> neighbors = closedNode.getNeighborsStack(terrainGenerator.lock());
		for (NodeAI& neighbor : neighbors)
		{
			if (isNodeClosedStack(neighbor))
				continue;
			
			neighbor.parentAddress = &(*std::prev(processedNodesStack.end()));

			const float distance = neighbor.distanceToEndPoint(endPos);
			const float terrainValue = terrainGenerator.lock()->getTerrainValue(neighbor.position.x, neighbor.position.y);
			float heuristicsValue = (1 - terrainValue) * (distance + neighbor.depth);

			nodesToProcessStack.emplace(heuristicsValue, std::move(neighbor));
		}
	}
	if (finishNode)
	{
		if (auto terrain_g = terrainGenerator.lock(); terrain_g != nullptr)
		{
			finishNode->drawReturnPathStack(terrain_g);
			finishNode->getPathStack(path);
			terrain_g->updateTerrain();
		}
	}
	
	nodesToProcessStack.clear();
	processedNodesStack.clear();
}

NodeAI ActorAI::getTheNearestNodeFromOpenStack()
{
	const auto node_it = std::begin(nodesToProcessStack);
	if (node_it != std::end(nodesToProcessStack))
	{
		NodeAI n = node_it->second;
		nodesToProcessStack.erase(node_it);
		return n;
	}
	return {};
}

bool ActorAI::isNodeClosedStack(const NodeAI& node)
{
	const auto& it = std::find_if(std::begin(processedNodesStack), std::end(processedNodesStack), [&node](const NodeAI& n)
	{
		return n.position == node.position;
	});
	return it != std::end(processedNodesStack);
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

}
