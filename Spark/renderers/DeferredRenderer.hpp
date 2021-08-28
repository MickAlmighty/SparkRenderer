#pragma once
#include <memory>

#include "Buffer.hpp"
#include "GBuffer.hpp"
#include "Renderer.hpp"
#include "lights/LightManager.h"
#include "ScreenQuad.hpp"
#include "effects/AmbientOcclusion.hpp"

namespace spark::resources
{
class Shader;
}  // namespace spark::resources

namespace spark::renderers
{
class DeferredRenderer : public Renderer
{
    public:
    DeferredRenderer(unsigned int width, unsigned int height);
    DeferredRenderer(const DeferredRenderer&) = delete;
    DeferredRenderer(DeferredRenderer&&) = delete;
    DeferredRenderer& operator=(const DeferredRenderer&) = delete;
    DeferredRenderer& operator=(DeferredRenderer&&) = delete;
    ~DeferredRenderer() override;

    protected:
    void renderMeshes(const std::shared_ptr<Scene>& scene) override;
    void resizeDerived(unsigned int width, unsigned int height) override;

    GLuint getDepthTexture() const override;
    GLuint getLightingTexture() const override;

    private:
    void createFrameBuffersAndTextures();

    GBuffer gBuffer;
    GLuint framebuffer{}, lightingTexture{};
    GLuint brdfLookupTexture{};
    ScreenQuad screenQuad{};
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
};
}  // namespace spark::renderers