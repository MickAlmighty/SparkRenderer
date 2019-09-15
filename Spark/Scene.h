#pragma once
#include <GameObject.h>
#include <memory>
#include <list>
#include <EngineSystems/SceneManager.h>
#include <Camera.h>

class GameObject;
class Component;
class Scene
{
private:
	friend class SceneManager;
	std::string name = "New Scene";
	std::shared_ptr<GameObject> root = std::make_shared<GameObject>("Root");
	std::weak_ptr<GameObject> gameObjectToPreview;
	std::shared_ptr<Camera>camera = std::make_shared<Camera>(glm::vec3(0, 0, 2));
public:
	Scene(std::string&& sceneName);
	~Scene();
	void update() const;
	void fixedUpdate() const;
	void removeGameObject(std::string&& name);
	void addGameObject(std::shared_ptr<GameObject> game_object);
	void addComponentToGameObject(std::shared_ptr<Component>& component, std::shared_ptr<GameObject> game_object);
	std::shared_ptr<Camera> getCamera() const;
	virtual void drawGUI();
	void drawSceneGraph();
	void drawTreeNode(std::shared_ptr<GameObject>& node);
};

