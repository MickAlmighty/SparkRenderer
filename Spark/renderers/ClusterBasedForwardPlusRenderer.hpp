#pragma once
#include "ClusterBasedLightCullingPass.hpp"
#include "profiling/GpuTimer.hpp"
#include "Renderer.hpp"
#include "utils/GlHandle.hpp"

namespace spark::renderers
{
class ClusterBasedForwardPlusRenderer : public Renderer
{
    public:
    ClusterBasedForwardPlusRenderer(unsigned int width, unsigned int height,
                                    const std::shared_ptr<resources::Shader>& activeClustersDeterminationShader,
                                    const std::shared_ptr<resources::Shader>& lightCullingShader);
    ClusterBasedForwardPlusRenderer(const ClusterBasedForwardPlusRenderer&) = delete;
    ClusterBasedForwardPlusRenderer(ClusterBasedForwardPlusRenderer&&) = delete;
    ClusterBasedForwardPlusRenderer& operator=(const ClusterBasedForwardPlusRenderer&) = delete;
    ClusterBasedForwardPlusRenderer& operator=(ClusterBasedForwardPlusRenderer&&) = delete;
    ~ClusterBasedForwardPlusRenderer() override;

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

    utils::TextureHandle brdfLookupTexture{}, lightingTexture{}, depthTexture{}, normalsTexture{};
    GLuint depthPrepassFramebuffer{}, lightingFramebuffer{};
    ScreenQuad screenQuad{};
    ClusterBasedLightCullingPass lightCullingPass;
    std::shared_ptr<resources::Shader> depthOnlyShader{nullptr};
    std::shared_ptr<resources::Shader> depthAndNormalsShader{nullptr};
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
    profiling::GpuTimer<3> timer;
};
}  // namespace spark::renderers