#include "WorldTransform.h"

#include <iostream>

#include <glm/gtx/euler_angles.hpp>

#include "JsonSerializer.h"

namespace spark {

WorldTransform::WorldTransform(glm::mat4 mat) : modelMatrix(mat) {}

glm::mat4 WorldTransform::getMatrix() const
{
	return modelMatrix;
}

glm::vec3 WorldTransform::getPosition() const
{
	return modelMatrix[3];
}

void WorldTransform::setMatrix(glm::mat4 mat)
{
	modelMatrix = mat;
	dirty = true;
}

void WorldTransform::setPosition(glm::vec3 position)
{
	modelMatrix[3] = glm::vec4(position, 1);
	dirty = true;
}

void WorldTransform::setPosition(float x, float y, float z)
{
	modelMatrix[3] = glm::vec4(x, y, z, 1);
	dirty = true;
}

void WorldTransform::translate(glm::vec3 translation)
{
	modelMatrix[3] = glm::vec4(getPosition() + translation, 1);
	dirty = true;
}

void WorldTransform::translate(float x, float y, float z)
{
	glm::vec3 position = getPosition();
	position.x += x;
	position.y += y;
	position.z += z;
	modelMatrix[3] = glm::vec4(position, 1);
	dirty = true;
}

void WorldTransform::setRotationRadians(glm::vec3& radians)
{
	const glm::vec3 degrees = glm::degrees(radians);
	modelMatrix = modelMatrix * glm::eulerAngleYXZ(degrees.y, degrees.x, degrees.z);
	dirty = true;
}

void WorldTransform::setRotationRadians(float x, float y, float z)
{
	const glm::vec3 degrees = glm::degrees(glm::vec3(x, y, z));
	modelMatrix = modelMatrix * glm::eulerAngleYXZ(degrees.y, degrees.x, degrees.z);
	dirty = true;
}

void WorldTransform::setRotationDegrees(glm::vec3& degrees)
{
	modelMatrix = modelMatrix * glm::eulerAngleYXZ(degrees.y, degrees.x, degrees.z);
	dirty = true;
}

void WorldTransform::setRotationDegrees(float x, float y, float z)
{
	modelMatrix = modelMatrix * glm::eulerAngleYXZ(y, x, z);
	dirty = true;
}

void WorldTransform::setScale(glm::vec3 scale)
{
	modelMatrix = glm::scale(modelMatrix, scale);
	dirty = true;
}

}

RTTR_REGISTRATION{
    rttr::registration::class_<spark::WorldTransform>("WorldTransform")
    .constructor()(rttr::policy::ctor::as_object)
    .property("modelMatrix", &spark::WorldTransform::modelMatrix);
}