#include "ActorAI.h"
#include <GameObject.h>
#include <iostream>
#include "TerrainGenerator.h"

void Node::drawReturnPath(std::vector<glm::vec3>& perlinValues) const
{
	std::cout << position.x << " " << position.y << std::endl;
	perlinValues[position.x * 20 + position.y].y = 1.0f;
	if (parent)
		parent->drawReturnPath(perlinValues);
}

void ActorAI::findPath()
{
	createMap();
	bool isPathFound = false;
	glm::vec2 finish = endPos;
	
	nodesToProcess.emplace(0, new Node(startPos));
	Node* finishNode = nullptr;
	while (!isPathFound)
	{
		if (nodesToProcess.empty())
		{
			std::cout << "Path hasn't been found" << std::endl;
			break;
		}

		Node* closedNode = getTheNearestNodeFromOpen();
		if(closedNode->position == static_cast<glm::vec2>(endPos))
		{
			finishNode = closedNode;
			break;
		}
		processedNodes.push_back(closedNode);
		std::list<Node*> neighbors = closedNode->getNeighbors();

		for (Node* neighbor : neighbors)
		{
			if (isNodeClosed(neighbor))
				continue;

			neighbor->parent = closedNode;
			float distance = neighbor->distanceToEndPoint(endPos);
			nodesToProcess.emplace(distance, neighbor);

		}
	}
	if(finishNode)
	{
		finishNode->drawReturnPath(getGameObject()->getComponent<TerrainGenerator>()->perlinValues);
		getGameObject()->getComponent<TerrainGenerator>()->updateTerrain();
	}
	nodesToProcess.clear();
	processedNodes.clear();
}

void ActorAI::createMap()
{
	for(int i = 0; i < 20; i++)
	{
		for(int j = 0; j < 20; j++)
		{
			map[i][j].position = { i, j };
		}
	}
}

Node* ActorAI::getTheNearestNodeFromOpen()
{
	auto node_it = std::begin(nodesToProcess);
	if(node_it != std::end(nodesToProcess))
	{
		Node* n = node_it->second;
		nodesToProcess.erase(node_it);
		return n;
	}
	return nullptr;
}

bool ActorAI::isNodeClosed(Node* node)
{
	const auto& it = std::find_if(std::begin(processedNodes), std::end(processedNodes), [&node](const Node* n)
	{
		return n->position == node->position;
	});
	if(it != std::end(processedNodes))
	{
		return true;
	}
	return false;
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
	if(ImGui::Button("FindPath"))
	{
		findPath();
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
