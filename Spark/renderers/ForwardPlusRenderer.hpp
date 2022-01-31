#pragma once
#include <memory>

#include "Renderer.hpp"
#include "profiling/GpuTimer.hpp"
#include "utils/GlHandle.hpp"

namespace spark::resources
{
class Shader;
}

namespace spark::renderers
{
class ForwardPlusRenderer : public Renderer
{
    public:
    ForwardPlusRenderer(unsigned int width, unsigned int height);
    ForwardPlusRenderer(const ForwardPlusRenderer&) = delete;
    ForwardPlusRenderer(ForwardPlusRenderer&&) = delete;
    ForwardPlusRenderer& operator=(const ForwardPlusRenderer&) = delete;
    ForwardPlusRenderer& operator=(ForwardPlusRenderer&&) = delete;
    ~ForwardPlusRenderer() override;

    protected:
    void renderMeshes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera) override;
    void resizeDerived(unsigned int width, unsigned int height) override;

    GLuint getDepthTexture() const override;
    GLuint getLightingTexture() const override;

    private:
    void depthPrepass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera);
    GLuint aoPass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera);
    void lightingPass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera, const GLuint ssaoTexture);
    void createFrameBuffersAndTextures();

    GLuint lightingFramebuffer{}, depthPrepassFramebuffer{};
    utils::TextureHandle brdfLookupTexture{}, lightingTexture{}, depthTexture{}, normalsTexture{};
    ScreenQuad screenQuad{};
    std::shared_ptr<resources::Shader> depthOnlyShader{nullptr};
    std::shared_ptr<resources::Shader> depthAndNormalsShader{nullptr};
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
    profiling::GpuTimer<2> timer;
};
}  // namespace spark::renderers