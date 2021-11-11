#include "LocalTransform.h"

#include "GUI/ImGui/imgui.h"

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace spark
{
LocalTransform::LocalTransform(glm::vec3 pos, glm::vec3 scale, glm::vec3 rotation) : position(pos), rotationEuler(rotation), scale(scale) {}

void LocalTransform::drawGUI()
{
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
    ImGui::SetNextWindowSizeConstraints(ImVec2(250, 100), ImVec2(FLT_MAX, 100));  // Width = 250, Height > 100
    ImGui::BeginChild(
        "Local Transform", {0, 0}, true,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_AlwaysAutoResize);
    if(ImGui::BeginMenuBar())
    {
        ImGui::Text("Transform");
        ImGui::EndMenuBar();
    }

    glm::vec3 oldPos = position;
    glm::vec3 oldScale = scale;
    glm::vec3 oldRotation = rotationEuler;

    ImGui::DragFloat3("Position", glm::value_ptr(position), 0.005f);
    ImGui::DragFloat3("Scale", glm::value_ptr(scale), 0.005f);
    ImGui::DragFloat3("Rotation", glm::value_ptr(rotationEuler), 0.1f);
    // setRotationDegrees(rotation);
    if(oldPos != position || oldScale != scale || oldRotation != rotationEuler)
        dirty = true;

    ImGui::EndChild();
    ImGui::PopStyleVar();
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
    const glm::mat4 posMatrix = glm::translate(glm::mat4(1), position);
    const glm::mat4 scaleMatrix = glm::scale(glm::mat4(1), scale);
    const glm::vec3 rotationRadians = glm::radians(rotationEuler);
    // const glm::mat4 rotationMatrix = glm::eulerAngleYXZ(roationRadians.y, roationRadians.x, roationRadians.z);
    const glm::mat4 rot = glm::mat4_cast(glm::quat(rotationRadians));
    return posMatrix * rot * scaleMatrix;
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
    return rotationEuler;
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
    rotationEuler.x = x;
    rotationEuler.y = y;
    rotationEuler.z = z;
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

}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::LocalTransform>("LocalTransform")
        .constructor()(rttr::policy::ctor::as_object)
        .property("position", &spark::LocalTransform::position)
        .property("rotationEuler", &spark::LocalTransform::rotationEuler)
        .property("scale", &spark::LocalTransform::scale)
        .property("modelMatrix", &spark::LocalTransform::modelMatrix);
}