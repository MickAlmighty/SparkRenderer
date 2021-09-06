#pragma once

#include <memory>

#include "Buffer.hpp"
#include "Camera.h"
#include "Scene.h"
#include "Shader.h"
#include "lights/LightManager.h"

namespace spark::renderers
{
class ClusterBasedLightCullingPass
{
    public:
    ClusterBasedLightCullingPass(unsigned int width, unsigned int height);
    ClusterBasedLightCullingPass(const ClusterBasedLightCullingPass&) = delete;
    ClusterBasedLightCullingPass(ClusterBasedLightCullingPass&&) = delete;
    ClusterBasedLightCullingPass& operator=(const ClusterBasedLightCullingPass&) = delete;
    ClusterBasedLightCullingPass& operator=(ClusterBasedLightCullingPass&&) = delete;
    ~ClusterBasedLightCullingPass() = default;

    void process(GLuint depthTexture, const std::shared_ptr<Scene>& scene);
    void resize(unsigned int width, unsigned int height);

    SSBO globalPointLightIndices{};
    SSBO globalSpotLightIndices{};
    SSBO globalLightProbeIndices{};
    SSBO perClusterGlobalLightIndicesBufferMetadata{};
    glm::vec2 pxTileSize{1};

    private:
    void createClusters(const std::shared_ptr<Scene>& scene);
    void determineActiveClusters(GLuint depthTexture, const std::shared_ptr<Scene>& scene);
    void buildCompactClusterList();
    void lightCulling(const std::shared_ptr<Scene>& scene);
    void clearActiveClustersCounter();

    unsigned int w{}, h{};

    float lastCamNearZ{-1.0f}, lastCamFarZ{-1.0f};
    glm::vec2 lastPxTileSize{1};

    const glm::uvec3 dispatchSize{64, 64, 32};

    SSBO clusters{};
    SSBO activeClusters{};
    SSBO activeClustersCount{12};
    SSBO activeClusterIndices{};

    std::shared_ptr<resources::Shader> clusterCreationShader{nullptr};
    std::shared_ptr<resources::Shader> determineActiveClustersShader{nullptr};
    std::shared_ptr<resources::Shader> buildCompactClusterListShader{nullptr};
    std::shared_ptr<resources::Shader> clusterBasedLightCullingShader{nullptr};
};
}  // namespace spark::renderers