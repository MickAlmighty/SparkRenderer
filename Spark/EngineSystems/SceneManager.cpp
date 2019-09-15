#include <EngineSystems/SceneManager.h>
#include <algorithm>
#include <EngineSystems/ResourceManager.h>
#include <Spark.h>
#include <EngineSystems/SceneManager.h>
#include <EngineSystems/ResourceManager.h>

std::shared_ptr<SceneManager> SceneManager::getInstance()
{
	static std::shared_ptr<SceneManager> scene_manager = nullptr;
	if(scene_manager == nullptr)
	{
		scene_manager = std::make_shared<SceneManager>();
	}
	return scene_manager;
}

void SceneManager::setup()
{
	auto scene = std::make_shared<Scene>("MainScene");
	std::shared_ptr<Component> model = ResourceManager::getInstance()->findModelMesh(Spark::pathToModelMeshes.string() + "\\box\\box.obj");
	auto gameObject = std::make_shared<GameObject>("FirstModel");
	scene->addComponentToGameObject(model, gameObject);
	scene->addGameObject(gameObject);

	std::shared_ptr<Component> model2 = ResourceManager::getInstance()->findModelMesh(Spark::pathToModelMeshes.string() + "\\box\\box.obj");
	auto gameObject2 = std::make_shared<GameObject>("SecondModel");
	scene->addComponentToGameObject(model2, gameObject2);
	gameObject2->transform.local.setPosition(-3, 2, -2);

	gameObject->addChild(gameObject2, gameObject);
	gameObject->transform.local.setRotationDegrees(45, 45, 0);
	//scene->addGameObject(gameObject2);


	addScene(scene);
	setCurrentScene("MainScene");

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
	
	if(scene_it != std::end(scenes))
	{
		current_scene = *scene_it;
		return true;
	}
	return false;
}

std::shared_ptr<Scene> SceneManager::getCurrentScene()
{
	return current_scene;
}
