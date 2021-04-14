#pragma once

#include <list>
#include <memory>

#include "Scene.h"

namespace spark
{
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
    bool setCurrentScene(const std::shared_ptr<Scene>& scene);
    std::shared_ptr<Scene> getCurrentScene() const;
    void drawGui();

    private:
    SceneManager() = default;
    ~SceneManager() = default;
    std::shared_ptr<Scene> current_scene = std::make_shared<Scene>();
};

}  // namespace spark