#pragma once

#include <memory>

#include "Scene.h"

namespace spark
{
class SceneManager final
{
    public:
    SceneManager();
    ~SceneManager() = default;
    SceneManager(const SceneManager&) = delete;
    SceneManager(SceneManager&&) = delete;
    SceneManager& operator=(const SceneManager&) = delete;
    SceneManager& operator=(SceneManager&&) = delete;

    void update() const;
    void fixedUpdate() const;
    bool setCurrentScene(const std::shared_ptr<Scene>& scene);
    std::shared_ptr<Scene> getCurrentScene() const;
    void drawGui();

    private:
    std::shared_ptr<Scene> current_scene = std::make_shared<Scene>();
};

}  // namespace spark