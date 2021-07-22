#include "Skybox.h"

#include "Spark.h"
#include "EngineSystems/SparkRenderer.h"

namespace spark
{
Skybox::Skybox() : Component("Skybox") {}

Skybox::~Skybox()
{
    if(isSkyboxActive())
        setActiveSkybox(false);
}

void Skybox::update() {}

void Skybox::fixedUpdate() {}

void Skybox::createPbrCubemap(const std::shared_ptr<resources::Texture>& texture)
{
    skyboxName = texture->getPath().filename().string();
    pbrCubemapTexture = std::make_shared<PbrCubemapTexture>(texture->getID());
    if(activeSkybox)
        SparkRenderer::getInstance()->setCubemap(pbrCubemapTexture);
}

void Skybox::drawGUI()
{
    ImGui::Text("Texture: ");
    ImGui::SameLine();
    ImGui::Text(skyboxName.c_str());
    bool active = isSkyboxActive();
    ImGui::Checkbox("IsSkyboxActive", &active);
    if(active != isSkyboxActive())
        setActiveSkybox(active);

    if(const auto tex = SparkGui::selectTextureByFilePicker(); tex)
    {
        if(const auto& texture = tex.value(); texture)
        {
            createPbrCubemap(texture);
        }
    }

    removeComponentGUI<Skybox>();
}

void Skybox::setActiveSkybox(bool isActive)
{
    if(isActive)
    {
        if(activeSkyboxPtr)
            activeSkyboxPtr->setActiveSkybox(false);

        activeSkyboxPtr = this;
        activeSkybox = true;
        SparkRenderer::getInstance()->setCubemap(pbrCubemapTexture);
    }
    else
    {
        if (activeSkyboxPtr == this)
        {
            SparkRenderer::getInstance()->setCubemap(nullptr);
            activeSkyboxPtr = nullptr;
            activeSkybox = false;
        }
    }
}

bool Skybox::isSkyboxActive() const
{
    return activeSkybox;
}

Skybox* Skybox::getActiveSkybox()
{
    return activeSkyboxPtr;
}

void Skybox::onActive()
{
    if(activeSkyboxPtr == this)
    {
        SparkRenderer::getInstance()->setCubemap(pbrCubemapTexture);
    }
}

void Skybox::onInactive()
{
    if(activeSkyboxPtr == this)
    {
        SparkRenderer::getInstance()->setCubemap(nullptr);
    }
}

std::string Skybox::getSkyboxName() const
{
    return skyboxName;
}

void Skybox::loadSkyboxByName(std::string name)
{
    skyboxName = std::move(name);
    const auto texture = Spark::resourceLibrary.getResourceByName<resources::Texture>(skyboxName);
    createPbrCubemap(texture);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::Skybox>("Skybox")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("skyboxName", &spark::Skybox::getSkyboxName, &spark::Skybox::loadSkyboxByName)
        .property("activeSkybox", &spark::Skybox::isSkyboxActive, &spark::Skybox::setActiveSkybox);
}
