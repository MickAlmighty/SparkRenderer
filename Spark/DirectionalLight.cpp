#include "DirectionalLight.h"

#include <glm/gtc/type_ptr.hpp>

#include "EngineSystems/SceneManager.h"
#include "GameObject.h"
#include "JsonSerializer.h"
#include "SerializerUtil.h"
#include "Structs.h"

namespace spark
{
DirectionalLightData DirectionalLight::getLightData() const
{
    return {direction, color * colorStrength};
}

bool DirectionalLight::getDirty() const
{
    return dirty;
}

glm::vec3 DirectionalLight::getDirection() const
{
    return direction;
}

glm::vec3 DirectionalLight::getColor() const
{
    return color;
}

float DirectionalLight::getColorStrength() const
{
    return colorStrength;
}

void DirectionalLight::resetDirty()
{
    dirty = false;
}

void DirectionalLight::setDirection(glm::vec3 direction_)
{
    dirty = true;
    direction = direction_;
}

void DirectionalLight::setColor(glm::vec3 color_)
{
    dirty = true;
    color = color_;
}

void DirectionalLight::setColorStrength(float strength)
{
    dirty = true;
    colorStrength = strength;
}

DirectionalLight::DirectionalLight() : Component("DirectionalLight") {}

void DirectionalLight::setActive(bool active_)
{
    dirty = true;
    active = active_;
}

void DirectionalLight::update()
{
    if(!addedToLightManager)
    {
        SceneManager::getInstance()->getCurrentScene()->lightManager->addDirectionalLight(shared_from_base<DirectionalLight>());
        addedToLightManager = true;
    }

    glm::vec3 lightDirection = getGameObject()->transform.local.getMatrix() * glm::vec4(dirLightFront, 0.0f);
    lightDirection = glm::normalize(lightDirection);
    if(lightDirection != direction)
    {
        setDirection(lightDirection);
    }
}

void DirectionalLight::fixedUpdate() {}

void DirectionalLight::drawGUI()
{
    glm::vec3 colorToEdit = getColor();
    glm::vec3 directionToEdit = getDirection();
    float colorStrengthToEdit = getColorStrength();
    ImGui::ColorEdit3("color", glm::value_ptr(colorToEdit));
    ImGui::DragFloat("colorStrength", &colorStrengthToEdit, 0.01f);
    ImGui::SliderFloat3("direction", glm::value_ptr(directionToEdit), -1.0f, 1.0f);

    if(colorStrengthToEdit < 0)
    {
        colorStrengthToEdit = 0;
    }

    if(directionToEdit != getDirection())
    {
        setDirection(directionToEdit);
    }
    if(colorToEdit != getColor())
    {
        setColor(colorToEdit);
    }
    if(colorStrengthToEdit != getColorStrength())
    {
        setColorStrength(colorStrengthToEdit);
    }
    removeComponentGUI<DirectionalLight>();
}

}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::DirectionalLight>("DirectionalLight")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        //.property("dirty", &spark::DirectionalLight::dirty) //FIXME: shouldn't it always be dirty when loaded? maybe not
        //.property("addedToLightManager", &spark::DirectionalLight::addedToLightManager)
        .property("color", &spark::DirectionalLight::color)
        .property("colorStrength", &spark::DirectionalLight::colorStrength)
        .property("dirLightFront", &spark::DirectionalLight::dirLightFront)
        (rttr::detail::metadata(SerializerMeta::Serializable, false))
        .property("direction", &spark::DirectionalLight::direction);
}