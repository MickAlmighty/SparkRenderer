#ifndef SCENE_H
#define SCENE_H

#include "LightManager.h"

#include <json/value.h>

#include <memory>
#include <list>
#include <functional>


namespace spark {

class GameObject;
class Component;
class Camera;
class Scene final : public std::enable_shared_from_this<Scene>
{
public:
	
	Scene(std::string&& sceneName);
	~Scene();
	Scene(Scene&) = delete;
	Scene(Scene&&) = delete;
	Scene& operator=(const Scene&) = delete;
	Scene& operator=(const Scene&&) = delete;

	void update();
	void fixedUpdate();
	void removeObjectsFromScene();
	std::shared_ptr<Camera> getCamera() const;
	void drawGUI();
	void drawSceneGraph();
private:
	friend class SceneManager;
	friend class Component;
	void drawTreeNode(std::shared_ptr<GameObject> node, bool isRootNode);
    std::list < std::function<void()> > toRemove;
    std::unique_ptr<LightManager> lightManager;
    std::shared_ptr<PbrCubemapTexture> cubemap;
	std::string name{ "New Scene" };
	std::shared_ptr<GameObject> root{};
	std::weak_ptr<GameObject> gameObjectToPreview;
	std::shared_ptr<Camera> camera{};
	bool cameraMovement { false };
};

}
#endif