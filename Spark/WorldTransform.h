#pragma once
#include <glm/glm.hpp>
#include <json/value.h>

class WorldTransform
{
	glm::mat4 modelMatrix = glm::mat4(1);
	bool dirty = true;
public:
	glm::mat4 getMatrix() const;
	glm::vec3 getPosition() const;
	Json::Value serialize() const;

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
	WorldTransform(glm::mat4 mat = glm::mat4(1));
	~WorldTransform() = default;
};

