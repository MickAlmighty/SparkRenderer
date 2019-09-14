#pragma once
#include <list>
#include <memory>
#include "Scene.h"

class Scene;
class SceneManager
{
	std::list<std::shared_ptr<Scene>> scenes;
	std::shared_ptr<Scene> current_scene = nullptr;
public:
	void setup();
	void update() const;
	void fixedUpdate() const;
	void cleanup();
	void addScene(const std::shared_ptr<Scene>& scene);
	bool setCurrentScene(std::string&& sceneName);

	static std::shared_ptr<SceneManager> getInstance();
	SceneManager() = default;
	~SceneManager() = default;
};

