#pragma once
#include <memory>

#include "Buffer.hpp"
#include "lights/LightManager.h"

namespace spark::resources
{
class Shader;
}
namespace spark::renderers
{
class TileBasedLightCullingPass
{
    public:
    TileBasedLightCullingPass(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                              const std::shared_ptr<lights::LightManager>& lightManager);
    TileBasedLightCullingPass(const TileBasedLightCullingPass&) = delete;
    TileBasedLightCullingPass(TileBasedLightCullingPass&&) = delete;
    TileBasedLightCullingPass& operator=(const TileBasedLightCullingPass&) = delete;
    TileBasedLightCullingPass& operator=(TileBasedLightCullingPass&&) = delete;
    ~TileBasedLightCullingPass();

    void process(GLuint depthTexture);
    void resize(unsigned int width, unsigned int height);

    void bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager);

    SSBO pointLightIndices{};
    SSBO spotLightIndices{};
    SSBO lightProbeIndices{};

    private:
    void createFrameBuffersAndTextures();

    unsigned int w{}, h{};
    GLuint lightsPerTileTexture{};
    std::shared_ptr<resources::Shader> tileBasedLightCullingShader{nullptr};
};
}  // namespace spark::renderers