#include "PointLight.h"

#include <map>

#include <glm/gtc/type_ptr.hpp>

#include "Enums.h"
#include "GameObject.h"
#include "JsonSerializer.h"
#include "Mesh.h"
#include "ResourceLoader.h"
#include "ShapeCreator.h"
#include "Spark.h"
#include "Structs.h"

namespace spark
{
PointLightData PointLight::getLightData() const
{
    return { glm::vec4(getPosition(), getRadius()), getColor() * getColorStrength(), getLightModel()};
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

float PointLight::getRadius() const
{
    return radius;
}

glm::mat4 PointLight::getLightModel() const
{
    return lightModel;
}

void PointLight::resetDirty()
{
    dirty = false;
}

void PointLight::setRadius(float radius)
{
    this->radius = radius;
    dirty = true;
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

void PointLight::setLightModel(glm::mat4 model)
{
    dirty = true;
    lightModel = model;
}

PointLight::PointLight() : Component("PointLight")
{
    std::filesystem::path p = Spark::pathToResources;

    const auto attribute = VertexShaderAttribute::createVertexShaderAttributeInfo(0, 3, ShapeCreator::createSphere(1.0f, 10));
    sphere = std::make_shared<Mesh>(std::vector<VertexShaderAttribute>{attribute}, 
        std::vector<unsigned int>{}, 
        std::map<TextureTarget, std::shared_ptr<resources::Texture>>{}, 
        "Mesh", 
        ShaderType::SOLID_COLOR_SHADER);
}

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

    glm::mat4 sphereModel(1);
    sphereModel = glm::scale(sphereModel, glm::vec3(radius));
    sphereModel[3] = glm::vec4(getPosition(), 1.0f);

    if (sphereModel != lightModel)
    {
        setLightModel(sphereModel);
        //it also takes light position into consideration
    }
    
    if (getGameObject() == getGameObject()->getScene()->getGameObjectToPreview())
    {
        sphere->addToRenderQueue(getLightModel());
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
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::PointLight>("PointLight")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
    //.property("dirty", &spark::PointLight::dirty) //FIXME: shouldn't it always be dirty when loaded? maybe not
    //.property("addedToLightManager", &spark::PointLight::addedToLightManager)
    .property("color", &spark::PointLight::color)
    .property("colorStrength", &spark::PointLight::colorStrength)
    .property("radius", &spark::PointLight::radius);
}