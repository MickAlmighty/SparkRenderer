#include "PointLight.h"

#include <map>

#include <glm/gtc/type_ptr.hpp>

#include "EngineSystems/SparkRenderer.h"
#include "Enums.h"
#include "GameObject.h"
#include "JsonSerializer.h"
#include "Mesh.h"
#include "RenderingRequest.h"
#include "ShapeCreator.h"
#include "Spark.h"
#include "Structs.h"

namespace spark
{
PointLightData PointLight::getLightData() const
{
    return {glm::vec4(getPosition(), getRadius()), getColor() * getColorStrength(), getLightModel()};
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

float PointLight::getRadius() const
{
    return radius;
}

glm::mat4 PointLight::getLightModel() const
{
    return lightModel;
}

void PointLight::setRadius(float radius)
{
    this->radius = radius;
    notifyAbout(LightCommand::update);
}

void PointLight::setColor(glm::vec3 color_)
{
    color = color_;
    notifyAbout(LightCommand::update);
}

void PointLight::setColorStrength(float strength)
{
    colorStrength = strength;
    notifyAbout(LightCommand::update);
}

void PointLight::setLightModel(glm::mat4 model)
{
    lightModel = model;
    notifyAbout(LightCommand::update);
}

PointLight::PointLight() : Component("PointLight")
{
    const auto attribute = VertexShaderAttribute::createVertexShaderAttributeInfo(0, 3, ShapeCreator::createSphere(1.0f, 10));
    auto vertexShaderAttributes = std::vector<VertexShaderAttribute>{attribute};
    auto indices = std::vector<unsigned int>{};
    auto textures = std::map<TextureTarget, std::shared_ptr<resources::Texture>>{};
    sphere = std::make_shared<Mesh>(vertexShaderAttributes, indices, textures, "Mesh", ShaderType::SOLID_COLOR_SHADER);
}

PointLight::~PointLight()
{
    notifyAbout(LightCommand::remove);
}

void PointLight::update()
{
    if(!lightManager)
    {
        lightManager = getGameObject()->getScene()->lightManager;
        add(lightManager);

        notifyAbout(LightCommand::add);
    }

    glm::mat4 sphereModel(1);
    sphereModel = glm::scale(sphereModel, glm::vec3(radius));
    sphereModel[3] = glm::vec4(getPosition(), 1.0f);

    if(sphereModel != lightModel)
    {
        setLightModel(sphereModel);
        // it also takes light position into consideration
    }

    if(getGameObject() == getGameObject()->getScene()->getGameObjectToPreview())
    {
        RenderingRequest request{};
        request.shaderType = sphere->shaderType;
        request.gameObject = getGameObject();
        request.mesh = sphere;
        request.model = sphereModel;

        SparkRenderer::getInstance()->addRenderingRequest(request);
    }
}

void PointLight::fixedUpdate() {}

void PointLight::drawGUI()
{
    glm::vec3 colorToEdit = getColor();
    float colorStrengthToEdit = getColorStrength();
    float r = radius;
    ImGui::ColorEdit3("color", glm::value_ptr(colorToEdit));
    ImGui::DragFloat("colorStrength", &colorStrengthToEdit, 0.01f);
    ImGui::DragFloat("radius", &r, 0.1f, 0);

    if(colorStrengthToEdit < 0)
    {
        colorStrengthToEdit = 0;
    }

    if(r < 0)
        r = 0;

    if(r != radius)
        setRadius(r);

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

void PointLight::onActive()
{
    notifyAbout(LightCommand::add);
}

void PointLight::onInactive()
{
    notifyAbout(LightCommand::remove);
}

void PointLight::notifyAbout(LightCommand command)
{
    const LightStatus<PointLight> status{command, this};
    notify(&status);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::PointLight>("PointLight")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("color", &spark::PointLight::getColor, &spark::PointLight::setColor)
        .property("colorStrength", &spark::PointLight::getColorStrength, &spark::PointLight::setColorStrength)
        .property("radius", &spark::PointLight::getRadius, &spark::PointLight::setRadius);
}