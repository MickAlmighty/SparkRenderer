#pragma once

#include <list>
#include <memory>

#include "Scene.h"
#include "Factory.h"

namespace spark
{
// class Scene;
class SceneManager final
{
    public:
    SceneManager(SceneManager&) = delete;
    SceneManager(SceneManager&&) = delete;
    SceneManager& operator=(const SceneManager&) = delete;
    SceneManager& operator=(SceneManager&&) = delete;

    static SceneManager* getInstance();

    void setup();
    void update() const;
    void fixedUpdate() const;
    void cleanup();
    void addScene(const std::shared_ptr<Scene>& scene);
    bool setCurrentScene(const std::string& sceneName);
    std::shared_ptr<Scene> getCurrentScene() const;
    void drawGui();

    private:
    SceneManager() = default;
    ~SceneManager() = default;
    std::list<std::shared_ptr<Scene>> scenes;
    std::shared_ptr<Scene> current_scene{Factory::createScene("MainScene")};
};

}  // namespace spark