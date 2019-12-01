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
		glm::ivec2 startPos{};
		glm::ivec2 endPos{};
		float movementSpeed = 1.0f;

		ActorAI(std::string&& newName = "ActorAI");
		~ActorAI();

		void setPath(const std::deque<glm::ivec2>& path_);
		void setPath(const std::vector<glm::ivec2>& path_);

		SerializableType getSerializableType() override;
		Json::Value serialize() override;
		void deserialize(Json::Value& root) override;
		void update() override;
		void fixedUpdate() override;
		void drawGUI() override;

	private:
		bool isTraveling = false;
		GLuint vao{};
		GLuint vbo{}, ebo{};
		std::deque<std::pair<bool, glm::ivec2>> path;

		void walkToEndOfThePath();
		void validateActorPosition(glm::vec3& position) const;
		void setStartPosition(glm::vec3& position);
		void randomizeEndPoint();

		void initPathMesh();
		int updatePathMesh(const std::deque<std::pair<bool, glm::ivec2>>& path) const;
	};
}
#endif