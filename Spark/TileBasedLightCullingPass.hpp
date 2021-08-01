#pragma once
#include <memory>

#include "Buffer.hpp"
#include "lights/LightManager.h"

namespace spark
{
namespace resources
{
    class Shader;
}

class TileBasedLightCullingPass
{
    public:
    TileBasedLightCullingPass() = default;
    TileBasedLightCullingPass(const TileBasedLightCullingPass&) = delete;
    TileBasedLightCullingPass(TileBasedLightCullingPass&&) = delete;
    TileBasedLightCullingPass& operator=(const TileBasedLightCullingPass&) = delete;
    TileBasedLightCullingPass& operator=(TileBasedLightCullingPass&&) = delete;
    ~TileBasedLightCullingPass();

    void setup(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo, const std::shared_ptr<lights::LightManager>& lightManager);
    void process(GLuint depthTexture);
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);
    void bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager);

    SSBO pointLightIndices{};
    SSBO spotLightIndices{};
    SSBO lightProbeIndices{};

    private:
    unsigned int w{}, h{};
    GLuint lightsPerTileTexture{};
    std::shared_ptr<resources::Shader> tileBasedLightCullingShader{nullptr};
};
}  // namespace spark