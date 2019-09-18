#pragma once
#include <GameObject.h>
#include <memory>
#include <list>
#include <EngineSystems/SceneManager.h>
#include <Camera.h>
#include <functional>
#include <json/value.h>
class GameObject;
class Component;
class Scene
{
private:
	friend class SceneManager;
	std::string name = "New Scene";
	std::shared_ptr<GameObject> root = std::make_shared<GameObject>("Root");
	std::weak_ptr<GameObject> gameObjectToPreview;
	std::shared_ptr<Camera> camera = std::make_shared<Camera>(glm::vec3(0, 0, 2));
public:
	std::list < std::function<void()> > toRemove;
	Scene(std::string&& sceneName);
	~Scene();
	void update();
	void fixedUpdate();
	void removeObjectsFromScene();
	void addGameObject(std::shared_ptr<GameObject> game_object);
	void addComponentToGameObject(std::shared_ptr<Component>& component, std::shared_ptr<GameObject> game_object);
	std::shared_ptr<Camera> getCamera() const;
	Json::Value serialize();
	virtual void drawGUI();
	void drawSceneGraph();
	void drawTreeNode(std::shared_ptr<GameObject> node, bool isRootNode);
};

