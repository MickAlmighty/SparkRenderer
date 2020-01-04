#ifndef WORLD_TRANSFORM_H
#define WORLD_TRANSFORM_H

#include <glm/glm.hpp>
#include <json/value.h>

namespace spark {

	class WorldTransform
	{
	public:
		bool dirty = true;

		WorldTransform(glm::mat4 mat = glm::mat4(1));
		~WorldTransform() = default;

		glm::mat4 getMatrix() const;
		glm::vec3 getPosition() const;
		
		void setMatrix(glm::mat4 mat);
		void setPosition(glm::vec3 position);
		void setPosition(float x, float y, float z);
		void translate(glm::vec3 translation);
		void translate(float x, float y, float z);
		void setRotationRadians(glm::vec3& radians);
		void setRotationRadians(float x, float y, float z);
		void setRotationDegrees(glm::vec3& degrees);
		void setRotationDegrees(float x, float y, float z);
		void setScale(glm::vec3 scale);

		Json::Value serialize() const;
		void deserialize(Json::Value& root);

	private:
		glm::mat4 modelMatrix = glm::mat4(1);
	};
}
#endif