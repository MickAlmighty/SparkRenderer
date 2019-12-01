#include "ActorAI.h"

#include <deque>
#include <iostream>

#include <glm/gtc/random.inl>

#include "Clock.h"
#include "CUDA/PathfindingManager.h"
#include "EngineSystems/SparkRenderer.h"
#include "GameObject.h"
#include "GUI/SparkGui.h"
#include "JsonSerializer.h"
#include "NodeAI.h"

namespace spark {

	void ActorAI::update()
	{
		glm::vec3 pos = getGameObject()->transform.world.getPosition();

		if (path.empty())
		{
			validateActorPosition(pos);
			setStartPosition(pos);
			randomizeEndPoint();
			PathFindingManager::getInstance()->addAgent(shared_from_base<ActorAI>());
		}
		else if (const bool firstNodeAchieved = path.begin()->first; firstNodeAchieved)
		{
			validateActorPosition(pos);
			setStartPosition(pos);
			randomizeEndPoint();
			PathFindingManager::getInstance()->addAgent(shared_from_base<ActorAI>());
		}

		if (!path.empty())
		{
			isTraveling = true;
		}

		if (isTraveling)
		{
			walkToEndOfThePath();
			const int indicesCount = updatePathMesh(path);
			if (indicesCount != 0)
			{
				const auto f = [shared_ptr = shared_from_base<ActorAI>(), indicesCount](std::shared_ptr<Shader>& shader)
				{
					glBindVertexArray(shared_ptr->vao);
					glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indicesCount), GL_UNSIGNED_INT, 0);
					glBindVertexArray(0);
				};
				SparkRenderer::getInstance()->renderQueue[ShaderType::PATH_SHADER].push_back(f);
			}
		}
	}

	void ActorAI::validateActorPosition(glm::vec3& position) const
	{
		const auto mapWidth = PathFindingManager::getInstance()->map.width - 1;
		const auto mapHeight = PathFindingManager::getInstance()->map.height - 1;
		if (position.x < 0) position.x = 0;
		if (position.z < 0) position.z = 0;
		if (position.x > mapWidth) position.x = static_cast<float>(mapWidth);
		if (position.z > mapHeight) position.z = static_cast<float>(mapHeight);
	}

	void ActorAI::setStartPosition(glm::vec3& position)
	{
		glm::vec2 pos{};
		if (position.x - static_cast<int>(position.x) < 0.5)
			pos.x = std::floor(position.x);
		else
			pos.x = std::ceil(position.x);

		if (position.z - static_cast<int>(position.z) < 0.5)
			pos.y = std::floor(position.z);
		else
			pos.y = std::ceil(position.z);

		startPos = static_cast<glm::ivec2>(pos);
	}

	void ActorAI::randomizeEndPoint()
	{
		do
		{
			const auto mapWidth = static_cast<float>(PathFindingManager::getInstance()->map.width);
			const auto mapHeight = static_cast<float>(PathFindingManager::getInstance()->map.height);
			endPos = { static_cast<int>(glm::linearRand(0.0f, mapWidth)), static_cast<int>(glm::linearRand(0.0f, mapHeight)) };
		} while (PathFindingManager::getInstance()->map.getTerrainValue(endPos.x, endPos.y) == 1.0f && startPos != endPos);
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

		const auto insertVertex = [&vertices, &indices](glm::vec3&& vertex)
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

		for (int i = -1; i < lineSegments; ++i)
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
					wayPoint.first = true;
					path.pop_front();
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
		return root;
	}

	void ActorAI::deserialize(Json::Value& root)
	{
		name = root.get("name", "ActorAI").asString();
		movementSpeed = root.get("movementSpeed", 1.0f).asFloat();
	}

	void ActorAI::drawGUI()
	{

		ImGui::DragInt2("startPos", &startPos.x);
		ImGui::DragInt2("endPos", &endPos.x);
		ImGui::DragFloat("movementSpeed", &movementSpeed, 0.1f);
		if (ImGui::Button("FindPath"))
		{
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

	void ActorAI::setPath(const std::deque<glm::ivec2>& path_)
	{
		path.clear();
		for (const glm::ivec2& wayPoint : path_)
		{
			path.emplace_back(std::make_pair(false, wayPoint));
		}
	}

	void ActorAI::setPath(const std::vector<glm::ivec2>& path_)
	{
		path.clear();
		for (const glm::ivec2& wayPoint : path_)
		{
			path.emplace_back(std::make_pair(false, wayPoint));
		}
	}
}
