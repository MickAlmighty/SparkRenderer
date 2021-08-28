#pragma once

#include <memory>

#include "Buffer.hpp"
#include "ClusterBasedLightCullingPass.hpp"
#include "GBuffer.hpp"
#include "glad_glfw3.h"
#include "Renderer.hpp"
#include "effects/AmbientOcclusion.hpp"
#include "lights/LightManager.h"

namespace spark
{
struct PbrCubemapTexture;
}

namespace spark::renderers
{
class ClusterBasedDeferredRenderer : public Renderer
{
    public:
    ClusterBasedDeferredRenderer(unsigned int width, unsigned int height);
    ClusterBasedDeferredRenderer(const ClusterBasedDeferredRenderer&) = delete;
    ClusterBasedDeferredRenderer(ClusterBasedDeferredRenderer&&) = delete;
    ClusterBasedDeferredRenderer& operator=(const ClusterBasedDeferredRenderer&) = delete;
    ClusterBasedDeferredRenderer& operator=(ClusterBasedDeferredRenderer&&) = delete;
    ~ClusterBasedDeferredRenderer() override;

    protected:
    void renderMeshes(const std::shared_ptr<Scene>& scene) override;
    void resizeDerived(unsigned int width, unsigned int height) override;

    GLuint getDepthTexture() const override;
    GLuint getLightingTexture() const override;

    private:
    void createFrameBuffersAndTextures();

    GLuint lightingTexture{};
    GLuint brdfLookupTexture{};
    GBuffer gBuffer;
    ClusterBasedLightCullingPass lightCullingPass;
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
};
}  // namespace spark::renderers