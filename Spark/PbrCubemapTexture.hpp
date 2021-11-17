#pragma once

#include <memory>

#include "glad_glfw3.h"

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
    ~PbrCubemapTexture();

    static GLuint createIrradianceCubemap(GLuint framebuffer, GLuint environmentCubemap, Cube& cube,
                                          const std::shared_ptr<resources::Shader>& irradianceShader);
    static GLuint createPreFilteredCubemap(GLuint framebuffer, GLuint environmentCubemap, unsigned int envCubemapSize, Cube& cube,
                                           const std::shared_ptr<resources::Shader>& prefilterShader,
                                           const std::shared_ptr<resources::Shader>& resampleCubemapShader);

    GLuint cubemap{};
    GLuint irradianceCubemap{};
    GLuint prefilteredCubemap{};

    private:
    void setup(GLuint hdrTexture, unsigned int cubemapSize);
    static GLuint createEnvironmentCubemapWithMipmapChain(GLuint framebuffer, GLuint equirectangularTexture, unsigned int size, Cube& cube,
                                                          const std::shared_ptr<resources::Shader>& equirectangularToCubemapShader);
    static GLuint createBrdfLookupTexture(GLuint framebuffer, unsigned int envCubemapSize, const std::shared_ptr<resources::Shader>& brdfShader);
    static GLuint createCubemapAndCopyDataFromFirstLayerOf(GLuint cubemap, unsigned int cubemapSize);
};
}  // namespace spark