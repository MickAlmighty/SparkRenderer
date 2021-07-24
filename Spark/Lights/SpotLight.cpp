#include "SpotLight.h"

#include <glm/gtc/type_ptr.hpp>

#include "GameObject.h"
#include "JsonSerializer.h"
#include "Structs.h"

namespace spark::lights
{
SpotLightData SpotLight::getLightData() const
{
    const auto getSoftCutOffAngleInRadians = [this] { return glm::radians((getOuterCutOff() - getOuterCutOff() * getSoftCutOffRatio()) * 0.5f); };

    SpotLightData data{};
    data.direction = getDirection();
    data.position = getPosition();
    data.color = getColor() * getColorStrength();
    data.cutOff = glm::cos(getSoftCutOffAngleInRadians());
    data.outerCutOff = glm::cos(glm::radians(getOuterCutOff() * 0.5f));
    data.maxDistance = getMaxDistance();
    data.boundingSphere = {getGameObject()->transform.world.getPosition(), maxDistance};
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

float SpotLight::getSoftCutOffRatio() const
{
    return softCutOffRatio;
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

void SpotLight::setSoftCutOffRatio(float softCutOffRatio_)
{
    if(softCutOffRatio < 0.0f)
    {
        softCutOffRatio = 0.0f;
    }
    else if(softCutOffRatio > 1.0f)
    {
        softCutOffRatio = 1.0f;
    }
    else
    {
        softCutOffRatio = softCutOffRatio_;
    }

    notifyAbout(LightCommand::update);
}

void SpotLight::setOuterCutOff(float outerCutOff_)
{
    if(outerCutOff_ < 0.0f)
    {
        outerCutOff = 0.0f;
    }
    else if(outerCutOff_ > 180.0f)
    {
        outerCutOff = 180.0f;
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

void SpotLight::update()
{
    if(!lightManager)
    {
        lightManager = getGameObject()->getScene()->lightManager;
        add(lightManager);

        notifyAbout(LightCommand::add);
    }

    static glm::vec3 pos{};
    if(pos != getPosition())
    {
        notifyAbout(LightCommand::update);
        pos = getPosition();
    }

    const glm::vec3 dir = glm::normalize(getGameObject()->transform.world.getMatrix() * glm::vec4(0.0f, -1.0f, 0.0f, 0.0f));
    if(dir != getDirection())
    {
        setDirection(dir);
    }
}

void SpotLight::fixedUpdate() {}

void SpotLight::drawGUI()
{
    glm::vec3 colorToEdit = getColor();
    float colorStrengthToEdit = getColorStrength();
    float softCutOffRatioToEdit = getSoftCutOffRatio();
    float outerCutOffToEdit = getOuterCutOff();
    glm::vec3 directionToEdit = getDirection();
    float maxDistanceToEdit = getMaxDistance();
    ImGui::ColorEdit3("color", glm::value_ptr(colorToEdit));
    ImGui::DragFloat("colorStrength", &colorStrengthToEdit, 0.01f);
    ImGui::DragFloat("softCutOffRatio", &softCutOffRatioToEdit, 0.01f, 0.0f, 1.0f);
    ImGui::DragFloat("outerCutOff", &outerCutOffToEdit, 1.0f, 0.0f, 180.0f);
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

    if(softCutOffRatioToEdit != getSoftCutOffRatio())
    {
        setSoftCutOffRatio(softCutOffRatioToEdit);
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

void SpotLight::onActive()
{
    notifyAbout(LightCommand::add);
}

void SpotLight::onInactive()
{
    notifyAbout(LightCommand::remove);
}

void SpotLight::notifyAbout(LightCommand command)
{
    const LightStatus<SpotLight> status{command, this};
    notify(&status);
}
}  // namespace spark::lights

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::lights::SpotLight>("SpotLight")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("color", &spark::lights::SpotLight::getColor, &spark::lights::SpotLight::setColor)
        .property("colorStrength", &spark::lights::SpotLight::getColorStrength, &spark::lights::SpotLight::setColorStrength)
        .property("direction", &spark::lights::SpotLight::getDirection, &spark::lights::SpotLight::setDirection)
        .property("softCutOffRatio", &spark::lights::SpotLight::getSoftCutOffRatio, &spark::lights::SpotLight::setSoftCutOffRatio)
        .property("outerCutOff", &spark::lights::SpotLight::getOuterCutOff, &spark::lights::SpotLight::setOuterCutOff)
        .property("maxDistance", &spark::lights::SpotLight::getMaxDistance, &spark::lights::SpotLight::setMaxDistance);
}