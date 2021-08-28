#pragma once
#include "ClusterBasedLightCullingPass.hpp"
#include "Renderer.hpp"

namespace spark::renderers
{
class ClusterBasedForwardPlusRenderer : public Renderer
{
    public:
    ClusterBasedForwardPlusRenderer(unsigned int width, unsigned int height);
    ClusterBasedForwardPlusRenderer(const ClusterBasedForwardPlusRenderer&) = delete;
    ClusterBasedForwardPlusRenderer(ClusterBasedForwardPlusRenderer&&) = delete;
    ClusterBasedForwardPlusRenderer& operator=(const ClusterBasedForwardPlusRenderer&) = delete;
    ClusterBasedForwardPlusRenderer& operator=(ClusterBasedForwardPlusRenderer&&) = delete;
    ~ClusterBasedForwardPlusRenderer() override;

    protected:
    void renderMeshes(const std::shared_ptr<Scene>& scene) override;
    void resizeDerived(unsigned int width, unsigned int height) override;

    GLuint getDepthTexture() const override;
    GLuint getLightingTexture() const override;

    private:
    void depthPrepass(const std::shared_ptr<Scene>& scene);
    GLuint aoPass(const std::shared_ptr<Scene>& scene);
    void lightingPass(const std::shared_ptr<Scene>& scene, const GLuint ssaoTexture);
    void createFrameBuffersAndTextures();

    GLuint lightingFramebuffer{}, lightingTexture{};
    GLuint depthPrepassFramebuffer{}, depthTexture{}, normalsTexture{};
    GLuint brdfLookupTexture{};
    ScreenQuad screenQuad{};
    ClusterBasedLightCullingPass lightCullingPass;
    std::shared_ptr<resources::Shader> depthOnlyShader{nullptr};
    std::shared_ptr<resources::Shader> depthAndNormalsShader{nullptr};
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
};
}  // namespace spark::renderers