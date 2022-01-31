#pragma once

#include <memory>

#include "GBuffer.hpp"
#include "glad_glfw3.h"
#include "profiling/GpuTimer.hpp"
#include "Renderer.hpp"
#include "TileBasedLightCullingPass.hpp"
#include "utils/GlHandle.hpp"

namespace spark
{
class PbrCubemapTexture;
}

namespace spark::renderers
{
class TileBasedDeferredRenderer : public Renderer
{
    public:
    TileBasedDeferredRenderer(unsigned int width, unsigned int height);
    TileBasedDeferredRenderer(const TileBasedDeferredRenderer&) = delete;
    TileBasedDeferredRenderer(TileBasedDeferredRenderer&&) = delete;
    TileBasedDeferredRenderer& operator=(const TileBasedDeferredRenderer&) = delete;
    TileBasedDeferredRenderer& operator=(TileBasedDeferredRenderer&&) = delete;
    ~TileBasedDeferredRenderer() override = default;

    protected:
    void renderMeshes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera) override;
    void resizeDerived(unsigned int width, unsigned int height) override;

    GLuint getDepthTexture() const override;
    GLuint getLightingTexture() const override;

    private:
    void createFrameBuffersAndTextures();
    void processLighting(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera, GLuint ssaoTexture);
    GLuint aoPass(const std::shared_ptr<ICamera>& camera);

    utils::TextureHandle brdfLookupTexture{}, lightingTexture{};
    GBuffer gBuffer;
    TileBasedLightCullingPass lightCullingPass;
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
    profiling::GpuTimer<3> timer;
};
}  // namespace spark::renderers