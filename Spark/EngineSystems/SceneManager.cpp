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

void SceneManager::setup()
{
    // const auto scene = Factory::createScene("MainScene");
    addScene(current_scene);
    // setCurrentScene("MainScene");
}

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
    scenes.clear();
}

void SceneManager::addScene(const std::shared_ptr<Scene>& scene)
{
    scenes.push_back(scene);
}

bool SceneManager::setCurrentScene(const std::string& sceneName)
{
    const auto searchingFunction = [&sceneName](const std::shared_ptr<Scene>& scene) { return scene->getName() == sceneName; };

    const auto scene_it = std::find_if(std::begin(scenes), std::end(scenes), searchingFunction);

    if(scene_it != std::end(scenes))
    {
        current_scene = *scene_it;
        SparkRenderer::getInstance()->setScene(current_scene);
        return true;
    }
    return false;
}

std::shared_ptr<Scene> SceneManager::getCurrentScene() const
{
    return current_scene;
}

void SceneManager::drawGui()
{
    if(ImGui::BeginMenu("SceneManager"))
    {
        /*if(ImGui::MenuItem("Save Current Scene"))
        {
            if(!JsonSerializer::getInstance()->saveSceneToFile(current_scene, "scene.json"))
            {
                SPARK_ERROR("Scene serialization failed!");
            }
        }*/
        if(ImGui::MenuItem("Load main scene"))
        {
            const std::shared_ptr<Scene> scene{JsonSerializer::getInstance()->loadSceneFromFile("scene.json")};
            if(scene != nullptr)
            {
                if(scene->getName() == current_scene->getName())
                {
                    scenes.remove(current_scene);
                    scenes.push_back(scene);
                    setCurrentScene(scene->getName());
                }
            }
        }
        ImGui::EndMenu();
    }

    current_scene->drawGUI();
}

}  // namespace spark