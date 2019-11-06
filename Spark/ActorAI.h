#ifndef ACTOR_AI_H
#define ACTOR_AI_H

#include <deque>

#include <glm/glm.hpp>

#include <Component.h>

namespace spark {
	
	class TerrainGenerator;
	class NodeAI;

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
		GLuint vao{};
		GLuint vbo{}, ebo{};

		std::multimap<float, std::shared_ptr<NodeAI>> nodesToProcess;
		std::multimap<float, NodeAI> nodesToProcessStack;
		
		std::list<std::shared_ptr<NodeAI>> processedNodes;
		std::list<NodeAI> processedNodesStack;

		std::deque<std::pair<bool, glm::ivec2>> path;
		std::weak_ptr<TerrainGenerator> terrainGenerator;

		void findPath();
		std::shared_ptr<NodeAI> getTheNearestNodeFromOpen();
		bool isNodeClosed(const std::shared_ptr<NodeAI>& node);
		void walkToEndOfThePath();

		void findPathStack();
		NodeAI getTheNearestNodeFromOpenStack();
		bool isNodeClosedStack(const NodeAI& node);
		void insertOrSwapNodeIntoOpenedList(float f, const NodeAI& node);

		void validateActorPosition(glm::vec3& position) const;
		void setStartPosition(glm::vec3& position);

		void initPathMesh();
		int updatePathMesh(const std::deque<std::pair<bool, glm::ivec2>>& path) const;
	};
}
#endif