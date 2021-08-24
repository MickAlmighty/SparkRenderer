#pragma once
#include "Renderer.hpp"
#include "TileBasedLightCullingPass.hpp"

namespace spark::renderers
{
class TileBasedForwardPlusRenderer : public Renderer
{
    public:
    TileBasedForwardPlusRenderer(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                                 const std::shared_ptr<lights::LightManager>& lightManager);
    TileBasedForwardPlusRenderer(const TileBasedForwardPlusRenderer&) = delete;
    TileBasedForwardPlusRenderer(TileBasedForwardPlusRenderer&&) = delete;
    TileBasedForwardPlusRenderer& operator=(const TileBasedForwardPlusRenderer&) = delete;
    TileBasedForwardPlusRenderer& operator=(TileBasedForwardPlusRenderer&&) = delete;
    ~TileBasedForwardPlusRenderer() override;

    GLuint process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                   const UniformBuffer& cameraUbo) override;
    void bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager) override;
    void resize(unsigned int width, unsigned int height) override;
    GLuint getDepthTexture() const override;

    private:
    void depthPrepass(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const UniformBuffer& cameraUbo);
    GLuint aoPass();
    void lightingPass(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                      const UniformBuffer& cameraUbo, const GLuint ssaoTexture);
    void createFrameBuffersAndTextures();

    GLuint lightingFramebuffer{}, lightingTexture{};
    GLuint depthPrepassFramebuffer{}, depthTexture{}, normalsTexture{};
    GLuint brdfLookupTexture{};
    ScreenQuad screenQuad{};
    TileBasedLightCullingPass lightCullingPass;
    std::shared_ptr<resources::Shader> depthOnlyShader{nullptr};
    std::shared_ptr<resources::Shader> depthAndNormalsShader{nullptr};
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
};
}  // namespace spark::renderers