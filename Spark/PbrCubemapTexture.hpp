#pragma once

#include <memory>

#include "glad_glfw3.h"
#include "utils/GlHandle.hpp"

namespace spark
{
struct Cube;

namespace resources
{
    class Shader;
};

class PbrCubemapTexture final
{
    public:
    PbrCubemapTexture(GLuint hdrTexture, unsigned int size = 1024);

    static utils::TextureHandle createIrradianceCubemap(GLuint framebuffer, utils::TextureHandle environmentCubemap, Cube& cube,
                                                        const std::shared_ptr<resources::Shader>& irradianceShader);
    static utils::TextureHandle createPreFilteredCubemap(GLuint framebuffer, utils::TextureHandle environmentCubemap, unsigned int envCubemapSize,
                                                         Cube& cube, const std::shared_ptr<resources::Shader>& prefilterShader,
                                                         const std::shared_ptr<resources::Shader>& resampleCubemapShader);

    utils::TextureHandle cubemap{};
    utils::TextureHandle irradianceCubemap{};
    utils::TextureHandle prefilteredCubemap{};

    private:
    void setup(GLuint hdrTexture, unsigned int cubemapSize);
    static utils::TextureHandle createEnvironmentCubemapWithMipmapChain(GLuint framebuffer, GLuint equirectangularTexture, unsigned int size,
                                                                        Cube& cube,
                                                                        const std::shared_ptr<resources::Shader>& equirectangularToCubemapShader);
    static utils::TextureHandle createBrdfLookupTexture(GLuint framebuffer, unsigned int envCubemapSize,
                                                        const std::shared_ptr<resources::Shader>& brdfShader);
    static utils::TextureHandle createCubemapAndCopyDataFromFirstLayerOf(utils::TextureHandle cubemap, unsigned int cubemapSize);
};
}  // namespace spark