#pragma once
#include <list>
#include <memory>
#include <Scene.h>

class Scene;
class SceneManager
{
	std::list<std::shared_ptr<Scene>> scenes;
	std::shared_ptr<Scene> current_scene = nullptr;
	void drawMainMenuGui();
	void drawSparkSettings(bool *p_open);
	int checkCurrentItem(const char** items) const;
public:
	void setup();
	void update() const;
	void fixedUpdate() const;
	void cleanup();
	void addScene(const std::shared_ptr<Scene>& scene);
	bool setCurrentScene(std::string&& sceneName);
	std::shared_ptr<Scene> getCurrentScene();
	static std::shared_ptr<SceneManager> getInstance();
	void drawGUI();
	SceneManager() = default;
	~SceneManager() = default;
};

