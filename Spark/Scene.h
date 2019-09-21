#ifndef SCENE_H
#define SCENE_H
#include <memory>
#include <list>
#include <functional>
#include <json/value.h>

class GameObject;
class Component;
class Camera;
class Scene
{
private:
	friend class SceneManager;
	std::string name = "New Scene";
	std::shared_ptr<GameObject> root{};
	std::weak_ptr<GameObject> gameObjectToPreview;
	std::shared_ptr<Camera> camera{};
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
	Json::Value serialize() const;
	void deserialize(Json::Value& root);
	virtual void drawGUI();
	void drawSceneGraph();
	void drawTreeNode(std::shared_ptr<GameObject> node, bool isRootNode);
};

#endif