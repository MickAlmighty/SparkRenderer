#include "Skybox.h"

#include "Spark.h"
#include "EngineSystems/SparkRenderer.h"

namespace spark
{
Skybox::Skybox() : Component("Skybox")
{
    skyBoxes.push_back(this);
}

Skybox::~Skybox()
{
    const auto it = std::find(skyBoxes.begin(), skyBoxes.end(), this);
    if(it != skyBoxes.end())
        skyBoxes.erase(it);
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
    if (active != isSkyboxActive())
        setActiveSkybox(active);

    const auto tex = SparkGui::getTexture();
    if(tex)
    {
        const auto& texture = tex.value();
        if(texture)
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
        const auto it =
            std::find_if(skyBoxes.cbegin(), skyBoxes.cend(), [this](Skybox* skybox) { return skybox != this && skybox->activeSkybox == true; });

        if(it != skyBoxes.end())
        {
            (*it)->setActiveSkybox(false);
        }
        activeSkybox = isActive;
        SparkRenderer::getInstance()->setCubemap(pbrCubemapTexture);
    }
    else
    {
        activeSkybox = isActive;
        SparkRenderer::getInstance()->setCubemap(nullptr);
    }
}

bool Skybox::isSkyboxActive() const
{
    return activeSkybox;
}

void Skybox::setActive(bool active_)
{
    if(!active_)
    {
        if(activeSkybox)
            SparkRenderer::getInstance()->setCubemap(nullptr);
    }
    else
    {
        if(activeSkybox)
            SparkRenderer::getInstance()->setCubemap(pbrCubemapTexture);
    }

    active = active_;
}

std::string Skybox::getSkyboxName() const
{
    return skyboxName;
}

void Skybox::loadSkyboxName(std::string name)
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
        .property("skyboxName", &spark::Skybox::getSkyboxName, &spark::Skybox::loadSkyboxName, rttr::registration::public_access)
        .property("activateSkybox", &spark::Skybox::isSkyboxActive, &spark::Skybox::setActiveSkybox, rttr::registration::public_access);
}
