#include "DirectionalLight.h"

#include <glm/gtc/type_ptr.hpp>

#include "Scene.h"
#include "GUI/ImGui/imgui.h"
#include "GameObject.h"
#include "JsonSerializer.h"

namespace spark::lights
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

DirectionalLight::~DirectionalLight()
{
    if(areLightShaftsEnabled())
    {
        deactivateLightShafts();
    }
    notifyAbout(LightCommand::remove);
}

void DirectionalLight::update()
{
    glm::vec3 lightDirection = getGameObject()->transform.local.getMatrix() * glm::vec4(dirLightFront, 0.0f);
    lightDirection = glm::normalize(lightDirection);
    if(lightDirection != direction)
    {
        setDirection(lightDirection);
    }
}

void DirectionalLight::drawUIBody()
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
    if(dirLightForLightShafts)
    {
        if(dirLightForLightShafts->getActive())
        {
            return dirLightForLightShafts;
        }
    }
    return nullptr;
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

void DirectionalLight::start()
{
    add(getGameObject()->getScene()->lightManager);

    notifyAbout(LightCommand::add);
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
}  // namespace spark::lights

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::lights::DirectionalLight>("DirectionalLight")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("color", &spark::lights::DirectionalLight::getColor, &spark::lights::DirectionalLight::setColor)
        .property("colorStrength", &spark::lights::DirectionalLight::getColorStrength, &spark::lights::DirectionalLight::setColorStrength)
        .property("direction", &spark::lights::DirectionalLight::getDirection, &spark::lights::DirectionalLight::setDirection)
        .property("lightShaftsActive", &spark::lights::DirectionalLight::areLightShaftsEnabled, &spark::lights::DirectionalLight::setLightShafts);
}