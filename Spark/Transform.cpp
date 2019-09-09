#include "Transform.h"
#include <glm/gtx/euler_angles.hpp>


Transform::Transform(glm::vec3 pos, glm::vec3 scale, glm::vec3 rotation)
{
	this->position = pos;
	this->scale = scale;
	this->rotationEuler = rotation;
}

Transform::~Transform()
{
}

glm::mat4 Transform::getMatrix() const
{
	glm::mat4 posMatrix = glm::translate(glm::mat4(1), position);
	glm::mat4 scaleMatrix = glm::scale(glm::mat4(1), scale);
	glm::mat4 rotationMatrix = glm::eulerAngleYXZ(rotationEuler.y, rotationEuler.x, rotationEuler.z);

	return posMatrix * rotationMatrix * scaleMatrix;
}

glm::vec3 Transform::getPosition() const
{
	return position;
}

glm::vec3 Transform::getScale() const
{
	return scale;
}

glm::vec3 Transform::getRotationRadians() const
{
	return glm::radians(rotationEuler);
}

glm::vec3 Transform::getRotationDegrees() const
{
	glm::vec3 degrees;
	return rotationEuler;
}

void Transform::setPosition(float x, float y, float z)
{
	position.x = x;
	position.t = y;
	position.z = z;
}

void Transform::setPosition(glm::vec3 pos)
{
	position = pos;
}

void Transform::setScale(float x, float y, float z)
{
	scale.x = x;
	scale.y = y;
	scale.z = z;
}

void Transform::setScale(glm::vec3 scale)
{
	this->scale = scale;
}

void Transform::setRotationDegrees(float x, float y, float z)
{
	rotationEuler = glm::vec3(x, y, z);
}

void Transform::setRotationDegrees(glm::vec3 rotationDegrees)
{
	rotationEuler = rotationDegrees;
}

void Transform::setRotationRadians(float x, float y, float z)
{
	rotationEuler = glm::degrees(glm::vec3(x, y, z));
}

void Transform::setRotationRadians(glm::vec3 rotationRadians)
{
	rotationEuler = glm::degrees(rotationRadians);
}
