#pragma once

#include <memory>

#include "Buffer.hpp"
#include "ClusterBasedLightCullingPass.hpp"
#include "GBuffer.h"
#include "glad_glfw3.h"
#include "Renderer.hpp"
#include "effects/AmbientOcclusion.hpp"
#include "lights/LightManager.h"

namespace spark
{
struct PbrCubemapTexture;

class ClusterBasedDeferredRenderer : public Renderer
{
    public:
    ClusterBasedDeferredRenderer(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                                 const std::shared_ptr<lights::LightManager>& lightManager);
    ClusterBasedDeferredRenderer(const ClusterBasedDeferredRenderer&) = delete;
    ClusterBasedDeferredRenderer(ClusterBasedDeferredRenderer&&) = delete;
    ClusterBasedDeferredRenderer& operator=(const ClusterBasedDeferredRenderer&) = delete;
    ClusterBasedDeferredRenderer& operator=(ClusterBasedDeferredRenderer&&) = delete;
    ~ClusterBasedDeferredRenderer() override;

    GLuint process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                   const UniformBuffer& cameraUbo) override;
    void resize(unsigned int width, unsigned int height) override;
    void bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager) override;

    GLuint getDepthTexture() const override;

    private:
    void createFrameBuffersAndTextures();

    GLuint lightingTexture{};
    GLuint brdfLookupTexture{};
    GBuffer gBuffer;
    ClusterBasedLightCullingPass lightCullingPass;
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
};
}  // namespace spark