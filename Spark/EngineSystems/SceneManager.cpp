#include "EngineSystems/SceneManager.h"

#include <algorithm>

#include <GUI/ImGui/imgui.h>

#include "EngineSystems/SparkRenderer.h"
#include "JsonSerializer.h"
#include "Scene.h"
#include "Spark.h"

namespace spark {

std::shared_ptr<SceneManager> SceneManager::getInstance()
{
	static std::shared_ptr<SceneManager> scene_manager = nullptr;
	if (scene_manager == nullptr)
	{
		scene_manager = std::make_shared<SceneManager>();
	}
	return scene_manager;
}

void SceneManager::setup()
{
	//const auto scene = std::make_shared<Scene>("MainScene");
	addScene(current_scene);
	//setCurrentScene("MainScene");
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

bool SceneManager::setCurrentScene(std::string&& sceneName)
{
	const auto searchingFunction = [&sceneName](const std::shared_ptr<Scene>& scene)
	{
		return scene->name == sceneName;
	};

	const auto scene_it = std::find_if(std::begin(scenes), std::end(scenes), searchingFunction);

	if (scene_it != std::end(scenes))
	{
		current_scene = *scene_it;
		return true;
	}
	return false;
}

std::shared_ptr<Scene> SceneManager::getCurrentScene() const
{
	return current_scene;
}

void SceneManager::drawGui() const
{
	if (ImGui::BeginMenu("SceneManager"))
	{
		std::string menuName = "Current Scene: " + current_scene->name;
		if (ImGui::BeginMenu(menuName.c_str()))
		{
			ImGui::MenuItem("Camera Movement", NULL, &current_scene->cameraMovement);
			const auto cubemapPtr = SparkGui::getCubemapTexture();
			if(cubemapPtr)
			{
				current_scene->cubemap = cubemapPtr;
			}
			ImGui::EndMenu();
		}
		if (ImGui::MenuItem("Save Current Scene"))
		{
			JsonSerializer::writeToFile("scene.json", current_scene->serialize());
		}
		if (ImGui::MenuItem("Load main scene"))
		{
			Json::Value root = JsonSerializer::readFromFile("scene.json");
			current_scene->deserialize(root);
		}
		ImGui::EndMenu();
	}

	current_scene->drawGUI();
}

}