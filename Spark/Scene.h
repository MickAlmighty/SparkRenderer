#ifndef SCENE_H
#define SCENE_H

#include "Structs.h"
#include "LightManager.h"

#include <rttr/registration_friend>
#include <rttr/registration>

#include <memory>
#include <list>
#include <functional>

namespace spark
{
class GameObject;
class Component;
class Camera;
class Scene final : public std::enable_shared_from_this<Scene>
{
    public:
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
    std::shared_ptr<LightManager> lightManager;  // FIXME: switch back to unique_ptr
    void setCubemapPath(const std::string path);
    std::string getCubemapPath() const;

    private:
    friend class SceneManager;
    friend class Factory;
    friend class SparkRenderer;
    Scene() = default;
    explicit Scene(std::string&& sceneName);
    void drawTreeNode(std::shared_ptr<GameObject> node, bool isRootNode);
    std::shared_ptr<GameObject> getGameObjectToPreview() const;
    void setGameObjectToPreview(const std::shared_ptr<GameObject> node);
    std::shared_ptr<PbrCubemapTexture> cubemap;
    std::string name{"New Scene"};
    std::shared_ptr<GameObject> root{};
    std::weak_ptr<GameObject> gameObjectToPreview;
    std::shared_ptr<Camera> camera{};
    bool cameraMovement{false};
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};

}  // namespace spark
#endif