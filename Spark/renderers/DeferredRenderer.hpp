#pragma once
#include <memory>

#include "GBuffer.hpp"
#include "profiling/GpuTimer.hpp"
#include "Renderer.hpp"
#include "ScreenQuad.hpp"
#include "utils/GlHandle.hpp"

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
    void renderMeshes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera) override;
    void resizeDerived(unsigned int width, unsigned int height) override;

    GLuint getDepthTexture() const override;
    GLuint getLightingTexture() const override;

    private:
    void createFrameBuffersAndTextures();
    GLuint processAo(const std::shared_ptr<ICamera>& camera);
    void processLighting(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera, GLuint aoTexture);

    GBuffer gBuffer;
    GLuint framebuffer{};
    utils::TextureHandle brdfLookupTexture{}, lightingTexture{};
    ScreenQuad screenQuad{};
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
    profiling::GpuTimer<2> timer;
};
}  // namespace spark::renderers