#pragma once

#include <memory>
#include <deque>
#include <functional>

#include <rttr/registration_friend>
#include <rttr/registration>

#include "Enums.h"
#include "Resource.h"
#include "Structs.h"
#include "lights/LightManager.h"
#include "renderers/RenderingRequest.h"

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
    ~Scene() override;
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
    std::shared_ptr<lights::LightManager> lightManager = std::make_unique<lights::LightManager>();
    std::shared_ptr<GameObject> getGameObjectToPreview() const;
    std::string getName() const;
    const std::map<ShaderType, std::deque<renderers::RenderingRequest>>& getRenderingQueues() const;
    const std::weak_ptr<PbrCubemapTexture> getSkyboxCubemap() const;
    void addRenderingRequest(const renderers::RenderingRequest& request);
    void setCubemap(const std::shared_ptr<PbrCubemapTexture>& cubemap);

    std::map<ShaderType, std::deque<renderers::RenderingRequest>> renderingQueues{};

    private:
    void init();
    void drawTreeNode(std::shared_ptr<GameObject> node, bool isRootNode);
    void setGameObjectToPreview(const std::shared_ptr<GameObject> node);
    void clearRenderQueues();

    std::shared_ptr<GameObject> root{};
    std::weak_ptr<GameObject> gameObjectToPreview;
    std::shared_ptr<Camera> camera;

    
    std::weak_ptr<PbrCubemapTexture> skyboxCubemap;

    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE()
};

}  // namespace spark