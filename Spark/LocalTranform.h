#ifndef LOCAL_TRANSFORM_H
#define LOCAL_TRANSFORM_H

#include <glm/glm.hpp>
#include <json/value.h>

namespace spark {

class LocalTransform
{
public:
	LocalTransform(glm::vec3 pos = glm::vec3(0), glm::vec3 scale = glm::vec3(1), glm::vec3 rotation = glm::vec3(0));
	~LocalTransform() = default;

	void drawGUI();
	Json::Value serialize() const;
	void deserialize(Json::Value& root);
	glm::mat4 getMatrix();
	glm::vec3 getPosition() const;
	glm::vec3 getScale() const;
	glm::vec3 getRotationRadians() const;
	glm::vec3 getRotationDegrees() const;

	void setPosition(float x, float y, float z);
	void setPosition(glm::vec3 pos);
	void translate(glm::vec3 translation);
	void translate(float x, float y, float z);
	void setScale(float x, float y, float z);
	void setScale(glm::vec3 scale);
	void setRotationDegrees(float x, float y, float z);
	void setRotationDegrees(glm::vec3 rotationDegrees);
	void setRotationRadians(float x, float y, float z);
	void setRotationRadians(glm::vec3 rotationRadians);

private:
	glm::vec3 position = glm::vec3(0);
	glm::vec3 rotationEuler = glm::vec3(0);
	glm::vec3 scale = glm::vec3(0);
	glm::mat4 modelMatrix = glm::mat4(1);
	bool dirty = true;
	glm::mat4 recreateMatrix() const;
};

}
#endif