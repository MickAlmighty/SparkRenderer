#include "PointLight.h"

#include "GameObject.h"
#include "Structs.h"
#include "JsonSerializer.h"
#include <glm/gtc/type_ptr.hpp>

namespace spark
{
PointLightData PointLight::getLightData() const
{
    return {getPosition(), getColor() * getColorStrength()};
}

bool PointLight::getDirty() const
{
    return dirty;
}

glm::vec3 PointLight::getPosition() const
{
    return getGameObject()->transform.world.getPosition();
}

glm::vec3 PointLight::getColor() const
{
    return color;
}

float PointLight::getColorStrength() const
{
    return colorStrength;
}

void PointLight::resetDirty()
{
    dirty = false;
}

void PointLight::setColor(glm::vec3 color_)
{
    dirty = true;
    color = color_;
}

void PointLight::setColorStrength(float strength)
{
    dirty = true;
    colorStrength = strength;
}

PointLight::PointLight() : Component("PointLight") {}

void PointLight::setActive(bool active_)
{
    dirty = true;
    active = active_;
}

void PointLight::update()
{
    if(!addedToLightManager)
    {
        SceneManager::getInstance()->getCurrentScene()->lightManager->addPointLight(shared_from_base<PointLight>());
        addedToLightManager = true;
    }

    const glm::vec3 newPos = getPosition();
    if(newPos != lastPos)
    {
        dirty = true;
    }
    lastPos = newPos;
}

void PointLight::fixedUpdate() {}

void PointLight::drawGUI()
{
    glm::vec3 colorToEdit = getColor();
    float colorStrengthToEdit = getColorStrength();
    ImGui::ColorEdit3("color", glm::value_ptr(colorToEdit));
    ImGui::DragFloat("colorStrength", &colorStrengthToEdit, 0.01f);

    if(colorStrengthToEdit < 0)
    {
        colorStrengthToEdit = 0;
    }

    if(colorToEdit != getColor())
    {
        setColor(colorToEdit);
    }
    if(colorStrengthToEdit != getColorStrength())
    {
        setColorStrength(colorStrengthToEdit);
    }
    removeComponentGUI<PointLight>();
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::PointLight>("PointLight")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        //.property("dirty", &spark::PointLight::dirty) //FIXME: shouldn't it always be dirty when loaded? maybe not
        //.property("addedToLightManager", &spark::PointLight::addedToLightManager)
        .property("color", &spark::PointLight::color)
        .property("colorStrength", &spark::PointLight::colorStrength)
        .property("lastPos", &spark::PointLight::lastPos);
}