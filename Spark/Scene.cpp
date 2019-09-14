#include "Scene.h"
#include <list>

Scene::Scene(std::string&& sceneName) : name(sceneName)
{

}

Scene::~Scene()
{
#ifdef DEBUG
	std::cout << "Scene destroyed!" << std::endl;
#endif
}

void Scene::update() const
{
	root->update();
}

void Scene::fixedUpdate() const
{
	root->fixedUpdate();
}

void Scene::removeGameObject(std::string&& name)
{

}

void Scene::addGameObject(std::shared_ptr<GameObject> game_object)
{
	root->addChild(game_object, root);
}

void Scene::addComponentToGameObject(std::shared_ptr<Component>& component, std::shared_ptr<GameObject> game_object)
{
	game_object->addComponent(component, game_object);
}
