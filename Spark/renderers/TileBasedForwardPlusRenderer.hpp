#pragma once
#include "profiling/GpuTimer.hpp"
#include "Renderer.hpp"
#include "TileBasedLightCullingPass.hpp"

namespace spark::renderers
{
class TileBasedForwardPlusRenderer : public Renderer
{
    public:
    TileBasedForwardPlusRenderer(unsigned int width, unsigned int height);
    TileBasedForwardPlusRenderer(const TileBasedForwardPlusRenderer&) = delete;
    TileBasedForwardPlusRenderer(TileBasedForwardPlusRenderer&&) = delete;
    TileBasedForwardPlusRenderer& operator=(const TileBasedForwardPlusRenderer&) = delete;
    TileBasedForwardPlusRenderer& operator=(TileBasedForwardPlusRenderer&&) = delete;
    ~TileBasedForwardPlusRenderer() override;

    protected:
    void resizeDerived(unsigned int width, unsigned int height) override;
    void renderMeshes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera) override;

    GLuint getDepthTexture() const override;
    GLuint getLightingTexture() const override;

    private:
    void depthPrepass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera);
    GLuint aoPass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera);
    void lightingPass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera, const GLuint ssaoTexture);
    void createFrameBuffersAndTextures();

    GLuint lightingFramebuffer{}, lightingTexture{};
    GLuint depthPrepassFramebuffer{}, depthTexture{}, normalsTexture{};
    GLuint brdfLookupTexture{};
    ScreenQuad screenQuad{};
    TileBasedLightCullingPass lightCullingPass;
    std::shared_ptr<resources::Shader> depthOnlyShader{nullptr};
    std::shared_ptr<resources::Shader> depthAndNormalsShader{nullptr};
    std::shared_ptr<resources::Shader> lightingShader{nullptr};

    profiling::GpuTimer<3> timer;
};
}  // namespace spark::renderers