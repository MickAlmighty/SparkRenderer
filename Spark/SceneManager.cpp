#include "SceneManager.h"

#include <imgui.h>

#include "JsonSerializer.h"
#include "ResourceLibrary.h"
#include "Scene.h"
#include "Spark.h"

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

    return success;
}

std::shared_ptr<Scene> SceneManager::getCurrentScene() const
{
    return current_scene;
}

void SceneManager::drawGui()
{
    if(ImGui::BeginMenu("SceneManager"))
    {
        if(const auto scenePath = SparkGui::getRelativePathToSaveSceneByFilePicker(); !scenePath.empty())
        {
            if(!JsonSerializer().saveSceneToFile(current_scene, scenePath))
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