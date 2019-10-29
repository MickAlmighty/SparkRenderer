#ifndef SCENE_H
#define SCENE_H

#include <memory>
#include <list>
#include <functional>

#include <json/value.h>

namespace spark {

class GameObject;
class Component;
class Camera;
class Scene
{
public:
	std::list < std::function<void()> > toRemove;
	
	Scene(std::string&& sceneName);
	~Scene();
	
	void update();
	void fixedUpdate();
	void removeObjectsFromScene();
	std::shared_ptr<Camera> getCamera() const;
	virtual void drawGUI();
	void drawSceneGraph();
	void drawTreeNode(std::shared_ptr<GameObject> node, bool isRootNode);

private:
	friend class SceneManager;
	std::string name = "New Scene";
	std::shared_ptr<GameObject> root{};
	std::weak_ptr<GameObject> gameObjectToPreview;
	std::shared_ptr<Camera> camera{};
	bool cameraMovement = false;
};

}
#endif