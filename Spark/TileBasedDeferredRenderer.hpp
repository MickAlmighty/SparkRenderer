#pragma once

#include <memory>

#include "Buffer.hpp"
#include "GBuffer.h"
#include "glad_glfw3.h"
#include "TileBasedLightCullingPass.hpp"
#include "effects/AmbientOcclusion.hpp"
#include "lights/LightManager.h"

namespace spark
{
struct PbrCubemapTexture;

class TileBasedDeferredRenderer
{
    public:
    TileBasedDeferredRenderer(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                              const std::shared_ptr<lights::LightManager>& lightManager);
    TileBasedDeferredRenderer(const TileBasedDeferredRenderer&) = delete;
    TileBasedDeferredRenderer(TileBasedDeferredRenderer&&) = delete;
    TileBasedDeferredRenderer& operator=(const TileBasedDeferredRenderer&) = delete;
    TileBasedDeferredRenderer& operator=(TileBasedDeferredRenderer&&) = delete;
    ~TileBasedDeferredRenderer();

    GLuint process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                   const UniformBuffer& cameraUbo);
    void resize(unsigned int width, unsigned int height);
    void bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager);

    GLuint getDepthTexture() const;

    bool isAmbientOcclusionEnabled{false};
    effects::AmbientOcclusion ao;

    private:
    void createFrameBuffersAndTextures();

    unsigned int w{}, h{};
    GLuint lightingTexture{};
    GLuint brdfLookupTexture{};
    GBuffer gBuffer;
    TileBasedLightCullingPass lightCullingPass;
    std::shared_ptr<resources::Shader> tileBasedLightingShader{nullptr};
};
}  // namespace spark