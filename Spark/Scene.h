#pragma once

#include <memory>
#include <list>
#include <functional>

#include <rttr/registration_friend>
#include <rttr/registration>

#include "Resource.h"
#include "Structs.h"
#include "Lights/LightManager.h"

namespace spark
{
class GameObject;
class Component;
class Camera;
class Scene final : public std::enable_shared_from_this<Scene>, public resourceManagement::Resource
{
    public:
    Scene();
    Scene(const std::filesystem::path& path_);
    Scene(const std::filesystem::path& path_, const std::shared_ptr<Scene>&& scene_);
    ~Scene();
    Scene(Scene&) = delete;
    Scene(Scene&&) = delete;
    Scene& operator=(const Scene&) = delete;
    Scene& operator=(const Scene&&) = delete;

    void update();
    void fixedUpdate();
    void removeObjectsFromScene();
    std::shared_ptr<Camera> getCamera() const;
    std::shared_ptr<GameObject> getRoot() const;
    void drawGUI();
    void drawSceneGraph();
    std::list<std::function<void()>> toRemove;
    std::shared_ptr<LightManager> lightManager;
    std::shared_ptr<GameObject> getGameObjectToPreview() const;
    std::string getName() const;

    private:
    void Init();
    void drawTreeNode(std::shared_ptr<GameObject> node, bool isRootNode);
    void setGameObjectToPreview(const std::shared_ptr<GameObject> node);
    std::shared_ptr<GameObject> root{};
    std::weak_ptr<GameObject> gameObjectToPreview;
    std::shared_ptr<Camera> camera{};

    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE()
};

}  // namespace spark