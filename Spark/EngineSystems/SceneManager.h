#ifndef SCENE_MANAGER_H
#define SCENE_MANAGER_H

#include <list>
#include <memory>

#include "Scene.h"

namespace spark {

//class Scene;
class SceneManager
{
public:
	~SceneManager() = default;
	SceneManager() = default;

	static std::shared_ptr<SceneManager> getInstance();

	void setup();
	void update() const;
	void fixedUpdate() const;
	void cleanup();
	void addScene(const std::shared_ptr<Scene>& scene);
	bool setCurrentScene(std::string&& sceneName);
	std::shared_ptr<Scene> getCurrentScene() const;
	void drawGui() const;
private:
	std::list<std::shared_ptr<Scene>> scenes;
	std::shared_ptr<Scene> current_scene = std::make_shared<Scene>("MainScene");
};

}

#endif