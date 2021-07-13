#include "DirectionalLight.h"

#include <glm/gtc/type_ptr.hpp>

#include "Scene.h"
#include "GameObject.h"
#include "JsonSerializer.h"
#include "ReflectionUtils.h"
#include "Structs.h"

namespace spark
{
DirectionalLightData DirectionalLight::getLightData() const
{
    return {direction, color * colorStrength};
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

void DirectionalLight::setDirection(glm::vec3 direction_)
{
    direction = direction_;
    notifyAbout(LightCommand::update);
}

void DirectionalLight::setColor(glm::vec3 color_)
{
    color = color_;
    notifyAbout(LightCommand::update);
}

void DirectionalLight::setColorStrength(float strength)
{
    colorStrength = strength;
    notifyAbout(LightCommand::update);
}

DirectionalLight::DirectionalLight() : Component("DirectionalLight") {}

DirectionalLight::~DirectionalLight()
{
    if(areLightShaftsEnabled())
    {
        dirLightForLightShafts = nullptr;
    }
    notifyAbout(LightCommand::remove);
}

void DirectionalLight::update()
{
    if(!lightManager)
    {
        lightManager = getGameObject()->getScene()->lightManager;
        add(lightManager);

        notifyAbout(LightCommand::add);
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
    bool areEnabled = areLightShaftsEnabled();
    ImGui::Checkbox("lightShafts", &areEnabled);
    ImGui::ColorEdit3("color", glm::value_ptr(colorToEdit));
    ImGui::DragFloat("colorStrength", &colorStrengthToEdit, 0.01f);
    ImGui::SliderFloat3("direction", glm::value_ptr(directionToEdit), -1.0f, 1.0f);

    if(areEnabled != areLightShaftsEnabled())
    {
        setLightShafts(areEnabled);
    }

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

bool DirectionalLight::areLightShaftsEnabled() const
{
    return lightShaftsActive;
}

void DirectionalLight::setLightShafts(bool state)
{
    if(state)
    {
        activateLightShafts();
    }
    else
    {
        deactivateLightShafts();
    }
}

DirectionalLight* DirectionalLight::getDirLightForLightShafts()
{
    return dirLightForLightShafts;
}

void DirectionalLight::activateLightShafts()
{
    if(dirLightForLightShafts)
        dirLightForLightShafts->deactivateLightShafts();

    dirLightForLightShafts = this;
    lightShaftsActive = true;
}

void DirectionalLight::deactivateLightShafts()
{
    if(dirLightForLightShafts == this)
    {
        dirLightForLightShafts = nullptr;
        lightShaftsActive = false;
    }
}

void DirectionalLight::onActive()
{
    notifyAbout(LightCommand::add);
}

void DirectionalLight::onInactive()
{
    notifyAbout(LightCommand::remove);
}

void DirectionalLight::notifyAbout(LightCommand command)
{
    const LightStatus<DirectionalLight> status{command, this};
    notify(&status);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::DirectionalLight>("DirectionalLight")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("color", &spark::DirectionalLight::getColor, &spark::DirectionalLight::setColor)
        .property("colorStrength", &spark::DirectionalLight::getColorStrength, &spark::DirectionalLight::setColorStrength)
        .property("direction", &spark::DirectionalLight::getDirection, &spark::DirectionalLight::setDirection)
        .property("lightShaftsActive", &spark::DirectionalLight::areLightShaftsEnabled, &spark::DirectionalLight::setLightShafts);
}