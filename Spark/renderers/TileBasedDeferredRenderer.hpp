#pragma once

#include <memory>

#include "Buffer.hpp"
#include "GBuffer.hpp"
#include "glad_glfw3.h"
#include "Renderer.hpp"
#include "TileBasedLightCullingPass.hpp"
#include "effects/AmbientOcclusion.hpp"
#include "lights/LightManager.h"

namespace spark
{
struct PbrCubemapTexture;
}

namespace spark::renderers
{
class TileBasedDeferredRenderer : public Renderer
{
    public:
    TileBasedDeferredRenderer(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                              const std::shared_ptr<lights::LightManager>& lightManager);
    TileBasedDeferredRenderer(const TileBasedDeferredRenderer&) = delete;
    TileBasedDeferredRenderer(TileBasedDeferredRenderer&&) = delete;
    TileBasedDeferredRenderer& operator=(const TileBasedDeferredRenderer&) = delete;
    TileBasedDeferredRenderer& operator=(TileBasedDeferredRenderer&&) = delete;
    ~TileBasedDeferredRenderer() override;

    GLuint process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                   const UniformBuffer& cameraUbo) override;
    void resize(unsigned int width, unsigned int height) override;
    void bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager) override;

    GLuint getDepthTexture() const override;

    private:
    void createFrameBuffersAndTextures();

    GLuint lightingTexture{};
    GLuint brdfLookupTexture{};
    GBuffer gBuffer;
    TileBasedLightCullingPass lightCullingPass;
    std::shared_ptr<resources::Shader> lightingShader{nullptr};
};
}  // namespace spark::renderers