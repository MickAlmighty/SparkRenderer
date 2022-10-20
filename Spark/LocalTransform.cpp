#include "LocalTransform.h"

#include <rttr/registration>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/orthonormalize.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Logging.h"
#include "GUI/imgui_custom_widgets.h"

namespace spark
{
LocalTransform::LocalTransform(glm::vec3 pos, glm::vec3 scale, glm::vec3 rot) : position(pos), rotation(rot), scale(scale) {}

void LocalTransform::drawGUI()
{
    ImGui::BeginGroupPanel("Local Transform", {-1, 0});

    glm::vec3 newPosition = position;
    glm::vec3 newScale = scale;
    glm::quat newRotation = rotation;

    ImGui::DragFloat3("Position", glm::value_ptr(newPosition), 0.005f);
    ImGui::DragFloat3("Scale", glm::value_ptr(newScale), 0.005f);
    ImGui::DragFloat4("Rotation", glm::value_ptr(newRotation), 0.1f);

    if(newRotation != rotation)
    {
        setRotation(newRotation);
    }
    if(newPosition != position)
    {
        setPosition(newPosition);
    }
    if(newScale != scale)
    {
        setScale(newScale);
    }

    ImGui::EndGroupPanel();
}

glm::mat4 LocalTransform::getMatrix()
{
    if(dirty)
    {
        modelMatrix = recreateMatrix();
        dirty = false;
        return modelMatrix;
    }
    return modelMatrix;
}

glm::mat4 LocalTransform::recreateMatrix() const
{
    const glm::mat4 positionMat = glm::translate(glm::mat4(1), position);
    const glm::mat4 scaleMat = glm::scale(glm::mat4(1), scale);
    const glm::mat4 rotationMat = glm::mat4_cast(glm::normalize(rotation));
    return positionMat * rotationMat * scaleMat;
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
    return glm::eulerAngles(rotation);
}

glm::vec3 LocalTransform::getRotationDegrees() const
{
    return glm::degrees(glm::eulerAngles(rotation));
}

glm::quat LocalTransform::getRotation() const
{
    return rotation;
}

void LocalTransform::setPosition(float x, float y, float z)
{
    position.x = x;
    position.y = y;
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
    rotation = glm::quat(glm::radians(glm::vec3(x, y, z)));
    dirty = true;
}

void LocalTransform::setRotationDegrees(glm::vec3 rotationDegrees)
{
    rotation = glm::quat(glm::radians(rotationDegrees));
    dirty = true;
}

void LocalTransform::setRotationRadians(float x, float y, float z)
{
    rotation = glm::quat(glm::vec3(x, y, z));
    dirty = true;
}

void LocalTransform::setRotationRadians(glm::vec3 rotationRadians)
{
    rotation = glm::quat(rotationRadians);
    dirty = true;
}

void LocalTransform::setRotation(glm::quat q)
{
    rotation = q;
    dirty = true;
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::LocalTransform>("LocalTransform")
        .constructor()(rttr::policy::ctor::as_object)
        .property("position", &spark::LocalTransform::position)
        .property("rotation", &spark::LocalTransform::rotation)
        .property("scale", &spark::LocalTransform::scale);
}