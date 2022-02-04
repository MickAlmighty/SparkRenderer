#include "Skybox.h"

#include "Spark.h"
#include "GameObject.h"
#include "Scene.h"

namespace spark
{
Skybox::~Skybox()
{
    if(isSkyboxActive())
        setActiveSkybox(false);
}

void Skybox::update() {}

void Skybox::createPbrCubemap(const std::shared_ptr<resources::Texture>& texture)
{
    skyboxPath = texture->getPath().string();
    pbrCubemapTexture = std::make_shared<PbrCubemapTexture>(texture->getID());

    if(isSkyboxActive())
    {
        setSkyboxInScene(pbrCubemapTexture);
    }
}

void Skybox::drawUIBody()
{
    ImGui::Text("Texture: ");
    ImGui::SameLine();
    ImGui::Text(skyboxPath.c_str());
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
}

void Skybox::setSkyboxInScene(const std::shared_ptr<PbrCubemapTexture>& skyboxCubemap) const
{
    if(const auto gameObject = getGameObject(); gameObject)
    {
        gameObject->getScene()->setCubemap(skyboxCubemap);
    }
}

void Skybox::setActiveSkybox(bool isActive)
{
    if(isActive)
    {
        if(const bool isThisSkyboxActive = isSkyboxActive(); isThisSkyboxActive)
        {
            return;
        }

        if(const bool isOtherSkyboxActive = activeSkyboxPtr != nullptr; isOtherSkyboxActive)
        {
            activeSkyboxPtr->setActiveSkybox(false);
        }

        activeSkyboxPtr = this;
        setSkyboxInScene(pbrCubemapTexture);
    }
    else
    {
        if(isSkyboxActive())
        {
            setSkyboxInScene(nullptr);
            activeSkyboxPtr = nullptr;
        }
    }
}

bool Skybox::isSkyboxActive() const
{
    return activeSkyboxPtr == this;
}

Skybox* Skybox::getActiveSkybox()
{
    return activeSkyboxPtr;
}

void Skybox::onActive()
{
    if(activeSkyboxPtr == this)
    {
        getGameObject()->getScene()->setCubemap(pbrCubemapTexture);
    }
}

void Skybox::onInactive()
{
    if(activeSkyboxPtr == this)
    {
        getGameObject()->getScene()->setCubemap(nullptr);
    }
}

std::string Skybox::getSkyboxPath() const
{
    return skyboxPath;
}

void Skybox::loadSkybox(std::string relativePath)
{
    skyboxPath = std::move(relativePath);
    const auto texture = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Texture>(skyboxPath);
    createPbrCubemap(texture);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::Skybox>("Skybox")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("skyboxPath", &spark::Skybox::getSkyboxPath, &spark::Skybox::loadSkybox)
        .property("activeSkybox", &spark::Skybox::isSkyboxActive, &spark::Skybox::setActiveSkybox);
}
