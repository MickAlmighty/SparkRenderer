#include <LocalTranform.h>
#include <glm/gtx/euler_angles.hpp>

LocalTransform::LocalTransform(glm::vec3 pos, glm::vec3 scale, glm::vec3 rotation)
{
	this->position = pos;
	this->scale = scale;
	this->rotationEuler = rotation;
}

glm::mat4 LocalTransform::getMatrix()
{
	if (dirty)
	{
		modelMatrix = recreateMatrix();
		dirty = false;
		return modelMatrix;
	}
	return modelMatrix;
}

glm::mat4 LocalTransform::recreateMatrix() const
{
	const glm::mat4 posMatrix = glm::translate(glm::mat4(1), position);
	const glm::mat4 scaleMatrix = glm::scale(glm::mat4(1), scale);
	const glm::mat4 rotationMatrix = glm::eulerAngleYXZ(rotationEuler.y, rotationEuler.x, rotationEuler.z);

	return posMatrix * rotationMatrix * scaleMatrix;
}

glm::vec3 LocalTransform::getPosition() const
{
	return position;
}

glm::vec3 LocalTransform::getScale() const
{
	return scale;
}

glm::vec3 LocalTransform::getRotationRadians() const
{
	return glm::radians(rotationEuler);
}

glm::vec3 LocalTransform::getRotationDegrees() const
{
	glm::vec3 degrees;
	return rotationEuler;
}

void LocalTransform::setPosition(float x, float y, float z)
{
	position.x = x;
	position.t = y;
	position.z = z;
	dirty = true;
}

void LocalTransform::setPosition(glm::vec3 pos)
{
	position = pos;
	dirty = true;
}

void LocalTransform::translate(glm::vec3 translation)
{
	position += translation;
	dirty = true;
}

void LocalTransform::translate(float x, float y, float z)
{
	position.x += x;
	position.y += y;
	position.z += z;
	dirty = true;
}

void LocalTransform::setScale(float x, float y, float z)
{
	scale.x = x;
	scale.y = y;
	scale.z = z;
	dirty = true;
}

void LocalTransform::setScale(glm::vec3 scale)
{
	this->scale = scale;
	dirty = true;
}

void LocalTransform::setRotationDegrees(float x, float y, float z)
{
	rotationEuler = glm::vec3(x, y, z);
	dirty = true;
}

void LocalTransform::setRotationDegrees(glm::vec3 rotationDegrees)
{
	rotationEuler = rotationDegrees;
	dirty = true;
}

void LocalTransform::setRotationRadians(float x, float y, float z)
{
	rotationEuler = glm::degrees(glm::vec3(x, y, z));
	dirty = true;
}

void LocalTransform::setRotationRadians(glm::vec3 rotationRadians)
{
	rotationEuler = glm::degrees(rotationRadians);
	dirty = true;
}

