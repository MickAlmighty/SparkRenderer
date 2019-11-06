#include "ActorAI.h"

#include <deque>
#include <iostream>

#include <glm/gtc/random.inl>

#include "Clock.h"
#include "EngineSystems/SparkRenderer.h"
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
		}

		const double measureStart = glfwGetTime();
		//findPath();
		findPathStack();
		timer = glfwGetTime() - measureStart;
		//std::cout << timer * 1000.0 << " ms" << std::endl;
		if (!path.empty())
		{
			isTraveling = true;
		}
		

		if (isTraveling)
		{
			walkToEndOfThePath();
			int indicesCount = updatePathMesh(path);
			const auto f = [shared_ptr = shared_from_base<ActorAI>(), indicesCount] (std::shared_ptr<Shader>& shader)
			{
				glBindVertexArray(shared_ptr->vao);
				glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indicesCount), GL_UNSIGNED_INT, 0);
				glBindVertexArray(0);
			};
			SparkRenderer::getInstance()->renderQueue[ShaderType::PATH_SHADER].push_back(f);
		}
	
		if (isTraveling)
		{
			walkToEndOfThePath();
			int indicesCount = updatePathMesh(path);
			const auto f = [this, indicesCount] (std::shared_ptr<Shader>& shader)
			{
				glBindVertexArray(vao);
				glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indicesCount), GL_UNSIGNED_INT, 0);
				glBindVertexArray(0);
			};
			SparkRenderer::getInstance()->renderQueue[ShaderType::PATH_SHADER].push_back(f);
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

	void ActorAI::initPathMesh()
	{
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<const void*>(0));

		glGenBuffers(1, &ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

		glBindVertexArray(0);
	}

	int ActorAI::updatePathMesh(const std::deque<std::pair<bool, glm::ivec2>>& path) const
	{
		if (path.size() < 2)
			return 0;
		const int numberOfVertices = 6;
		const int lineSegments = static_cast<int>(path.size() - 1);
		std::vector<glm::vec3> vertices;
		std::vector<unsigned int> indices;

		const auto insertVertex = [&vertices, &indices] (glm::vec3&& vertex)
		{
			const auto vertexIt = std::find(vertices.begin(), vertices.end(), vertex);
			if (vertexIt != vertices.end())
			{
				indices.push_back(static_cast<unsigned int>(std::distance(vertices.begin(), vertexIt)));
			}
			else
			{
				vertices.push_back(vertex);
				indices.push_back(static_cast<unsigned int>(vertices.size() - 1));
			}
		};

		for(int i = -1; i < lineSegments; ++i)
		{
			glm::vec3 segmentStart;
			if (i == -1)
			{
				segmentStart = getGameObject()->transform.world.getPosition();
			}
			else
			{
				segmentStart = { path[i].second.x, 0.0f, path[i].second.y };
			}
			glm::vec3 segmentEnd = { path[i + 1].second.x, 0.0f, path[i + 1].second.y };

			glm::vec3 direction = glm::normalize(segmentEnd - segmentStart);
			glm::vec3 perpendicularToDirection = glm::cross(direction, glm::vec3(0.0f, 1.0f, 0.0f));
		
			insertVertex(segmentStart + perpendicularToDirection * 0.1f);
			insertVertex(segmentEnd - perpendicularToDirection * 0.1f);
			insertVertex(segmentStart - perpendicularToDirection * 0.1f);
			insertVertex(segmentStart + perpendicularToDirection * 0.1f);
			insertVertex(segmentEnd + perpendicularToDirection * 0.1f);
			insertVertex(segmentEnd - perpendicularToDirection * 0.1f);
		}

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		return static_cast<int>(indices.size());
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

				const float functionH = neighbor->measureManhattanDistance(endPos);
				const float functionG = neighbor->depth = neighbor->measureManhattanDistance(startPos);
				const float terrainValue = terrainGenerator.lock()->getTerrainValue(neighbor->position.x, neighbor->position.y);
				float heuristicsValue = (1 - terrainValue) * (functionH + functionG);
				
				nodesToProcess.emplace(heuristicsValue, neighbor);
			}
		}
		if (finishNode)
		{
			if (auto terrain_g = terrainGenerator.lock(); terrain_g != nullptr)
			{
				path.clear();
				//finishNode->drawReturnPath(terrain_g);
				finishNode->getPath(path);
				path.pop_front();
				//terrain_g->updateTerrain();
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
		{
			isTraveling = false;
			return;
		}
			

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
				if (glm::distance(position, pointOnPath) < 0.1f)
				{
					//terrainGenerator.lock()->unMarkNodeAsPartOfPath(wayPoint.second.x, wayPoint.second.y);
					//terrainGenerator.lock()->updateTerrain();
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

				const float functionH = neighbor.measureManhattanDistance(endPos);
				const float functionG = neighbor.depth;
				const float terrainValue = terrainGenerator.lock()->getTerrainValue(neighbor.position.x, neighbor.position.y);
				const float heuristicsValue = (1 - terrainValue) * (functionH + functionG);

				insertOrSwapNodeIntoOpenedList(heuristicsValue, neighbor);
			}
		}
		if (finishNode)
		{
			if (auto terrain_g = terrainGenerator.lock(); terrain_g != nullptr)
			{
				//finishNode->drawReturnPathStack(terrain_g);
				finishNode->getPathStack(path);
				path.pop_front();
				//terrain_g->updateTerrain();
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

	void ActorAI::insertOrSwapNodeIntoOpenedList(float f, const NodeAI& node)
	{
		const auto& it = std::find_if(std::begin(nodesToProcessStack), std::end(nodesToProcessStack), [&node](const std::pair<float, NodeAI>& n)
		{
			const bool nodesEqual = n.second.position == node.position;
			if (!nodesEqual)
			{
				return false;
			}
			const bool betterFunctionG = node.depth < n.second.depth;
			
			return nodesEqual && betterFunctionG;
		});

		if (it != std::end(nodesToProcessStack))
		{
			nodesToProcessStack.erase(it);
			nodesToProcessStack.emplace(f, node);
		}
		else
		{
			nodesToProcessStack.emplace(f, node);
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

	void ActorAI::drawGUI()
	{
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
	}

	ActorAI::ActorAI(std::string&& newName) : Component(newName)
	{
		initPathMesh();
	}

	ActorAI::~ActorAI()
	{
		glDeleteBuffers(1, &vbo);
		glDeleteBuffers(1, &ebo);
		glDeleteVertexArrays(1, &vao);
	}

}
