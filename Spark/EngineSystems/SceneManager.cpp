#include "EngineSystems/SceneManager.h"

#include <algorithm>

#include <GUI/ImGui/imgui.h>

#include "EngineSystems/SparkRenderer.h"
#include "JsonSerializer.h"
#include "Scene.h"
#include "Spark.h"

namespace spark
{
SceneManager* SceneManager::getInstance()
{
    static SceneManager sceneManager{};
    return &sceneManager;
}

void SceneManager::setup() {}

void SceneManager::update() const
{
    current_scene->update();
}

void SceneManager::fixedUpdate() const
{
    current_scene->fixedUpdate();
}

void SceneManager::cleanup()
{
    current_scene = nullptr;
}

bool SceneManager::setCurrentScene(const std::shared_ptr<Scene>& scene)
{
    if(scene)
    {
        current_scene = scene;
        SparkRenderer::getInstance()->setScene(current_scene);
        return true;
    }

    current_scene = std::make_shared<Scene>();
    SparkRenderer::getInstance()->setScene(current_scene);
    return false;
}

std::shared_ptr<Scene> SceneManager::getCurrentScene() const
{
    return current_scene;
}

std::optional<std::shared_ptr<Scene>> getScene()
{
    bool objectPicked{false};
    std::shared_ptr<Scene> scene{nullptr};

    if(ImGui::BeginPopupModal("Scenes", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        const auto sceneIds = Spark::resourceLibrary.getSceneResourceIdentifiers();
        for(const auto& id : sceneIds)
        {
            if(ImGui::Button(id->getFullPath().string().c_str()))
            {
                scene = std::static_pointer_cast<Scene>(id->getResource());
                objectPicked = true;
                ImGui::CloseCurrentPopup();
            }
        }

        if(ImGui::Button("Close"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if(objectPicked)
        return {scene};

    return std::nullopt;
}

void SceneManager::drawGui()
{
    if(ImGui::BeginMenu("SceneManager"))
    {
        if(ImGui::MenuItem("Save Current Scene"))
        {
            if(!JsonSerializer::getInstance()->saveSceneToFile(current_scene, current_scene->getPath()))
            {
                SPARK_ERROR("Scene serialization failed!");
            }
        }
        if(ImGui::Button("Load main scene"))
        {
            ImGui::OpenPopup("Scenes");
        }

        const auto sceneOpt = getScene();
        if(sceneOpt.has_value())
        {
            if(sceneOpt.value() != nullptr)
            {
                setCurrentScene(sceneOpt.value());
            }
        }

        ImGui::EndMenu();
    }

    current_scene->drawGUI();
}

}  // namespace spark