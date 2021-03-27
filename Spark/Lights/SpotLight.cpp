#include "SpotLight.h"

#include <glm/gtc/type_ptr.hpp>

#include "GameObject.h"
#include "JsonSerializer.h"
#include "Structs.h"

namespace spark
{
using Status = LightStatus<SpotLight>;

SpotLightData SpotLight::getLightData() const
{
    SpotLightData data{};
    data.direction = getDirection();
    data.position = getPosition();
    data.color = getColor() * getColorStrength();
    data.cutOff = glm::cos(glm::radians(getCutOff()));
    data.outerCutOff = glm::cos(glm::radians(getOuterCutOff()));
    data.maxDistance = getMaxDistance();
    data.boundingSphere = { getGameObject()->transform.world.getPosition(), maxDistance };
    return data;
}

glm::vec3 SpotLight::getPosition() const
{
    return getGameObject()->transform.world.getPosition();
}

glm::vec3 SpotLight::getDirection() const
{
    return direction;
}

glm::vec3 SpotLight::getColor() const
{
    return color;
}

float SpotLight::getColorStrength() const
{
    return colorStrength;
}

float SpotLight::getCutOff() const
{
    return cutOff;
}

float SpotLight::getOuterCutOff() const
{
    return outerCutOff;
}

float SpotLight::getMaxDistance() const
{
    return maxDistance;
}

void SpotLight::setColor(glm::vec3 color_)
{
    color = color_;
    notifyAbout(LightCommand::update);
}

void SpotLight::setColorStrength(float strength)
{
    colorStrength = strength;
    notifyAbout(LightCommand::update);
}

void SpotLight::setDirection(glm::vec3 direction_)
{
    direction = direction_;
    notifyAbout(LightCommand::update);
}

void SpotLight::setCutOff(float cutOff_)
{
    if (cutOff_ < 0.0f)
    {
        cutOff = 0.0f;
    }
    else if (cutOff_ > 90.0f)
    {
        cutOff = 90.0f;
    }
    else
    {
        cutOff = cutOff_;
    }

    notifyAbout(LightCommand::update);
}

void SpotLight::setOuterCutOff(float outerCutOff_)
{
    if(outerCutOff_ < 0.0f)
    {
        outerCutOff = 0.0f;
    }
    else if(outerCutOff_ > 90.0f)
    {
        outerCutOff = 90.0f;
    }
    else
    {
        outerCutOff = outerCutOff_;
    }

    notifyAbout(LightCommand::update);
}

void SpotLight::setMaxDistance(float maxDistance_)
{
    maxDistance = maxDistance_;
    notifyAbout(LightCommand::update);
}

SpotLight::SpotLight() : Component("SpotLight") {}

SpotLight::~SpotLight()
{
    notifyAbout(LightCommand::remove);
}

void SpotLight::setActive(bool active_)
{
    active = active_;
    if(active)
    {
        notifyAbout(LightCommand::add);
    }
    else
    {
        notifyAbout(LightCommand::remove);
    }
}

void SpotLight::update()
{
    if(!lightManager)
    {
        lightManager = getGameObject()->getScene()->lightManager;
        add(lightManager);

        notifyAbout(LightCommand::add);
    }

    const glm::vec3 newPos = getPosition();
    if(newPos != lastPos)
    {
        notifyAbout(LightCommand::update);
    }
    lastPos = newPos;
}

void SpotLight::fixedUpdate() {}

void SpotLight::drawGUI()
{
    glm::vec3 colorToEdit = getColor();
    float colorStrengthToEdit = getColorStrength();
    float cutOffToEdit = getCutOff();
    float outerCutOffToEdit = getOuterCutOff();
    glm::vec3 directionToEdit = getDirection();
    float maxDistanceToEdit = getMaxDistance();
    ImGui::ColorEdit3("color", glm::value_ptr(colorToEdit));
    ImGui::DragFloat("colorStrength", &colorStrengthToEdit, 0.01f);
    ImGui::DragFloat("cutOff", &cutOffToEdit, 1.0f, 0.0f, 90.0f);
    ImGui::DragFloat("outerCutOff", &outerCutOffToEdit, 1.0f, 0.0f, 90.0f);
    ImGui::SliderFloat3("direction", glm::value_ptr(directionToEdit), -1.0f, 1.0f);
    ImGui::DragFloat("maxDistance", &maxDistanceToEdit, 0.1f, 0.0f);

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

    if(cutOffToEdit != getCutOff())
    {
        setCutOff(cutOffToEdit);
    }

    if(outerCutOffToEdit != getOuterCutOff())
    {
        setOuterCutOff(outerCutOffToEdit);
    }

    if(directionToEdit != getDirection())
    {
        setDirection(directionToEdit);
    }

    if(maxDistanceToEdit < 0)
    {
        maxDistanceToEdit = 0;
    }

    if(maxDistanceToEdit != getMaxDistance())
    {
        setMaxDistance(maxDistanceToEdit);
    }

    removeComponentGUI<SpotLight>();
}

void SpotLight::notifyAbout(LightCommand command)
{
    const LightStatus<SpotLight> status{command, this};
    notify(&status);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::SpotLight>("SpotLight")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("color", &spark::SpotLight::color)
        .property("colorStrength", &spark::SpotLight::colorStrength)
        .property("direction", &spark::SpotLight::direction)
        .property("cutOff", &spark::SpotLight::cutOff)
        .property("outerCutOff", &spark::SpotLight::outerCutOff)
        .property("lastPos", &spark::SpotLight::lastPos)
        .property("maxDistance", &spark::SpotLight::maxDistance);
}