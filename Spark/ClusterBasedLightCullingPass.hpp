#pragma once
#include <memory>

#include "Buffer.hpp"
#include "Camera.h"
#include "Shader.h"
#include "lights/LightManager.h"

namespace spark
{
class ClusterBasedLightCullingPass
{
    public:
    ClusterBasedLightCullingPass(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                                 const std::shared_ptr<lights::LightManager>& lightManager);
    ClusterBasedLightCullingPass(const ClusterBasedLightCullingPass&) = delete;
    ClusterBasedLightCullingPass(ClusterBasedLightCullingPass&&) = delete;
    ClusterBasedLightCullingPass& operator=(const ClusterBasedLightCullingPass&) = delete;
    ClusterBasedLightCullingPass& operator=(ClusterBasedLightCullingPass&&) = delete;
    ~ClusterBasedLightCullingPass() = default;

    void process(GLuint depthTexture);
    void resize(unsigned int width, unsigned int height);

    void bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager);

    SSBO globalPointLightIndices{};
    SSBO globalSpotLightIndices{};
    SSBO globalLightProbeIndices{};
    SSBO perClusterGlobalLightIndicesBufferMetadata{};
    glm::vec2 pxTileSize{ 1 };

    private:
    unsigned int w{}, h{};
    
    const glm::uvec3 dispatchSize{64, 64, 32};

    SSBO clusters{};
    SSBO activeClusters{};
    SSBO activeClustersCount{12};
    SSBO activeClusterIndices{};
    SSBO globalLightIndicesOffset{};

    std::shared_ptr<resources::Shader> clusterCreationShader{nullptr};
    std::shared_ptr<resources::Shader> determineActiveClustersShader{nullptr};
    std::shared_ptr<resources::Shader> buildCompactClusterListShader{nullptr};
    std::shared_ptr<resources::Shader> clusterBasedLightCullingShader{nullptr};
};
}  // namespace spark