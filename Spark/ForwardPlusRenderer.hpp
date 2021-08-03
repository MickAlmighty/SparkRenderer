#pragma once
#include <memory>

#include "Buffer.hpp"
#include "Enums.h"
#include "Renderer.hpp"
#include "RenderingRequest.h"
#include "effects/AmbientOcclusion.hpp"
#include "lights/LightManager.h"

namespace spark
{
namespace resources
{
    class Shader;
}

class ForwardPlusRenderer : public Renderer
{
    public:
    ForwardPlusRenderer(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                        const std::shared_ptr<lights::LightManager>& lightManager);
    ForwardPlusRenderer(const ForwardPlusRenderer&) = delete;
    ForwardPlusRenderer(ForwardPlusRenderer&&) = delete;
    ForwardPlusRenderer& operator=(const ForwardPlusRenderer&) = delete;
    ForwardPlusRenderer& operator=(ForwardPlusRenderer&&) = delete;
    ~ForwardPlusRenderer() override;

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
    std::shared_ptr<resources::Shader> depthOnlyShader{nullptr};
    std::shared_ptr<resources::Shader> depthAndNormalsShader{nullptr};
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
};
}  // namespace spark