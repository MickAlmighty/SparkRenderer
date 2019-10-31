#ifndef SCENE_MANAGER_H
#define SCENE_MANAGER_H

#include <list>
#include <memory>

#include "Scene.h"
#include "Factory.h"

namespace spark {

//class Scene;
    class SceneManager final {
    public:
        ~SceneManager() = default;
        SceneManager() = default;
        SceneManager(SceneManager&) = delete;
        SceneManager(SceneManager&&) = delete;
        SceneManager& operator=(const SceneManager&) = delete;
        SceneManager& operator=(SceneManager&&) = delete;

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
        std::shared_ptr<Scene> current_scene{ Factory::createScene("MainScene") };
    };

}

#endif