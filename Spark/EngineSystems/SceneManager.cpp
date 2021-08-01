#include "SceneManager.h"

#include "GUI/ImGui/imgui.h"

#include "JsonSerializer.h"
#include "ResourceLibrary.h"
#include "Scene.h"
#include "Spark.h"
#include "SparkRenderer.h"

namespace spark
{
SceneManager::SceneManager()
{
    current_scene = std::make_shared<Scene>();
}

void SceneManager::update() const
{
    current_scene->update();
}

void SceneManager::fixedUpdate() const
{
    current_scene->fixedUpdate();
}

bool SceneManager::setCurrentScene(const std::shared_ptr<Scene>& scene)
{
    bool success{false};
    if(scene)
    {
        current_scene = scene;
        success = true;
    }
    else
    {
        current_scene = std::make_shared<Scene>();
    }

    Spark::get().getRenderer().setScene(current_scene);
    return success;
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
        const auto sceneIds = Spark::get().getResourceLibrary().getSceneResourceIdentifiers();
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
        if(const auto scenePath = SparkGui::getRelativePathToSaveSceneByFilePicker(); !scenePath.empty())
        {
            if(!JsonSerializer::getInstance()->saveSceneToFile(current_scene, scenePath))
            {
                SPARK_ERROR("Scene serialization failed!");
            }
        }

        if(const auto sceneOpt = SparkGui::selectSceneByFilePicker(); sceneOpt.has_value())
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