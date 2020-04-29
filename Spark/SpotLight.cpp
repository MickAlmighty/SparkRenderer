#include "SpotLight.h"

#include <glm/gtc/type_ptr.hpp>

#include "GameObject.h"
#include "JsonSerializer.h"
#include "Structs.h"

namespace spark
{
SpotLightData SpotLight::getLightData() const
{
    SpotLightData data{};
    data.direction = getDirection();
    data.position = getPosition();
    data.color = getColor() * getColorStrength();
    data.cutOff = glm::cos(glm::radians(getCutOff()));
    data.outerCutOff = glm::cos(glm::radians(getOuterCutOff()));
    data.maxDistance = getMaxDistance();
    data.boundingSphere = calculateCullingSphereProperties();
    //SPARK_INFO("Spot light sphere pos {}, {}, {}, radius: {}",data.boundingSphere.x, data.boundingSphere.y, data.boundingSphere.z, data.boundingSphere.w);
    return data;
}

bool SpotLight::getDirty() const
{
    return dirty;
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

void SpotLight::resetDirty()
{
    dirty = false;
}

void SpotLight::setColor(glm::vec3 color_)
{
    dirty = true;
    color = color_;
}

void SpotLight::setColorStrength(float strength)
{
    dirty = true;
    colorStrength = strength;
}

void SpotLight::setDirection(glm::vec3 direction_)
{
    dirty = true;
    direction = direction_;
}

void SpotLight::setCutOff(float cutOff_)
{
    if(cutOff_ < 0.0f)
        return;
    if(cutOff_ > 360.0f)
        return;

    dirty = true;
    cutOff = cutOff_;
}

void SpotLight::setOuterCutOff(float outerCutOff_)
{
    dirty = true;
    outerCutOff = outerCutOff_;
}

void SpotLight::setMaxDistance(float maxDistance_)
{
    dirty = true;
    maxDistance = maxDistance_;
}

SpotLight::SpotLight() : Component("SpotLight") {}

void SpotLight::setActive(bool active_)
{
    dirty = true;
    active = active_;
}

void SpotLight::update()
{
    if(!addedToLightManager)
    {
        getGameObject()->getScene()->lightManager->addSpotLight(shared_from_base<SpotLight>());
        addedToLightManager = true;
    }

    const glm::vec3 newPos = getPosition();
    if(newPos != lastPos)
    {
        dirty = true;
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
    ImGui::DragFloat("cutOff", &cutOffToEdit, 1.0f, 0.0f, 180.0f);
    ImGui::DragFloat("outerCutOff", &outerCutOffToEdit, 1.0f, 0.0f, 180.0f);
    ImGui::SliderFloat3("direction", glm::value_ptr(directionToEdit), -1.0f, 1.0f);
    ImGui::DragFloat("maxDistance", &maxDistanceToEdit, 1.0f, 0.0f);

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

glm::vec4 SpotLight::calculateCullingSphereProperties() const
{
    /*if (outerCutOff == 180)
    {
        return {getGameObject()->transform.world.getPosition(), maxDistance};
    }*/

    const float angleRad = glm::radians(outerCutOff);
    const glm::vec3 pos = getGameObject()->transform.world.getPosition();
    /*if (angleRad > glm::pi<float>() / 4.0f)
    {
        const float radius = glm::tan(angleRad) * maxDistance;
        const glm::vec3 center = pos + direction * radius;
        return {center, radius};
    }
    else
    {
        const float radius = maxDistance * 0.5f / glm::pow(cos(angleRad), 2.0f);
        const glm::vec3 center = pos + direction * radius;
        return {center, radius};
    }*/

    /*if (angleRad > glm::pi<float>() / 4.0f)
    {
        const glm::vec3 center = pos + cos(angleRad) * maxDistance * direction;
        const float radius = sin(angleRad) * maxDistance;
        return {center, radius};
    }
    else
    {
        const glm::vec3 center = pos + maxDistance / (2.0f * cos(angleRad)) * direction;
        const float radius = maxDistance / (2.0f * cos(angleRad));
        return {center, radius};
    }*/

    return glm::vec4(pos, maxDistance);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::SpotLight>("SpotLight")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        //.property("dirty", &spark::SpotLight::dirty) //FIXME: shouldn't it always be dirty when loaded? maybe not
        //.property("addedToLightManager", &spark::SpotLight::addedToLightManager)
        .property("color", &spark::SpotLight::color)
        .property("colorStrength", &spark::SpotLight::colorStrength)
        .property("direction", &spark::SpotLight::direction)
        .property("cutOff", &spark::SpotLight::cutOff)
        .property("lastPos", &spark::SpotLight::lastPos)
        .property("maxDistance", &spark::SpotLight::maxDistance);
}