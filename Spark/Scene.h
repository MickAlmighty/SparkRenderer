#pragma once
#include "GameObject.h"
#include <memory>
#include <list>
#include "SceneManager.h"

class GameObject;
class Component;
class Scene
{
private:
	friend class SceneManager;
	std::string name = "New Scene";
	std::shared_ptr<GameObject> root = std::make_shared<GameObject>("Root");
public:
	Scene(std::string&& sceneName);
	~Scene();
	void update() const;
	void fixedUpdate() const;
	void removeGameObject(std::string&& name);
	void addGameObject(std::shared_ptr<GameObject> game_object);
	void addComponentToGameObject(std::shared_ptr<Component>& component, std::shared_ptr<GameObject> game_object);
};

