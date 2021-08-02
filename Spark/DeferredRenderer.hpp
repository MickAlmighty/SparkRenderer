#pragma once
#include <memory>

#include "Buffer.hpp"
#include "GBuffer.h"
#include "lights/LightManager.h"
#include "ScreenQuad.hpp"
#include "effects/AmbientOcclusion.hpp"

namespace spark
{
namespace resources
{
    class Shader;
}

class DeferredRenderer
{
    public:
    DeferredRenderer(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
        const std::shared_ptr<lights::LightManager>& lightManager);
    DeferredRenderer(const DeferredRenderer&) = delete;
    DeferredRenderer(DeferredRenderer&&) = delete;
    DeferredRenderer& operator=(const DeferredRenderer&) = delete;
    DeferredRenderer& operator=(DeferredRenderer&&) = delete;
    ~DeferredRenderer();

    GLuint process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                   const UniformBuffer& cameraUbo);
    void bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager);
    void resize(unsigned int width, unsigned int height);
    GLuint getDepthTexture() const;

    bool isAmbientOcclusionEnabled{false};
    effects::AmbientOcclusion ao;

    private:
    void createFrameBuffersAndTextures();

    unsigned int w{}, h{};
    GBuffer gBuffer;
    GLuint framebuffer{}, lightingTexture{};
    GLuint brdfLookupTexture{};
    ScreenQuad screenQuad{};
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
};
}  // namespace spark