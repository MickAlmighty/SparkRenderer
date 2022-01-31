#pragma once
#include <memory>

#include "Buffer.hpp"
#include "ICamera.hpp"
#include "utils/GlHandle.hpp"

namespace spark
{
namespace lights
{
    class LightManager;
}
namespace resources
{
    class Shader;
}
class Scene;
}  // namespace spark

namespace spark::renderers
{
class TileBasedLightCullingPass
{
    public:
    TileBasedLightCullingPass(unsigned int width, unsigned int height);
    TileBasedLightCullingPass(const TileBasedLightCullingPass&) = delete;
    TileBasedLightCullingPass(TileBasedLightCullingPass&&) = delete;
    TileBasedLightCullingPass& operator=(const TileBasedLightCullingPass&) = delete;
    TileBasedLightCullingPass& operator=(TileBasedLightCullingPass&&) = delete;
    ~TileBasedLightCullingPass() = default;

    void process(GLuint depthTexture, const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera);
    void resize(unsigned int width, unsigned int height);

    SSBO pointLightIndices{};
    SSBO spotLightIndices{};
    SSBO lightProbeIndices{};

    private:
    void bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager);
    void createFrameBuffersAndTextures();

    unsigned int w{}, h{};
    utils::TextureHandle lightsPerTileTexture{};
    std::shared_ptr<resources::Shader> tileBasedLightCullingShader{nullptr};
};
}  // namespace spark::renderers