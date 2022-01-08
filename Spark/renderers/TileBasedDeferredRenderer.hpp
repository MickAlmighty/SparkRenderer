#pragma once

#include <memory>

#include "GBuffer.hpp"
#include "glad_glfw3.h"
#include "Renderer.hpp"
#include "TileBasedLightCullingPass.hpp"

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
    ~TileBasedDeferredRenderer() override;

    protected:
    void renderMeshes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera) override;
    void resizeDerived(unsigned int width, unsigned int height) override;

    GLuint getDepthTexture() const override;
    GLuint getLightingTexture() const override;

    private:
    void createFrameBuffersAndTextures();

    GLuint lightingTexture{};
    GLuint brdfLookupTexture{};
    GBuffer gBuffer;
    TileBasedLightCullingPass lightCullingPass;
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
};
}  // namespace spark::renderers