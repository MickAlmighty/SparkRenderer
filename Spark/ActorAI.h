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
		inline void update() override;
		void fixedUpdate() override;
		void drawGUI() override;

	private:
		bool isTraveling = false;
		GLuint vao{};
		GLuint vbo{}, ebo{};
		std::deque<std::pair<bool, glm::ivec2>> path;

		inline void walkToEndOfThePath();
		inline void validateActorPosition(glm::vec3& position) const;
		inline void setStartPosition(glm::vec3& position);
		inline void randomizeEndPoint();

		void initPathMesh();
		inline int updatePathMesh(const std::deque<std::pair<bool, glm::ivec2>>& path) const;
	};
}
#endif