#pragma once

#include <memory>

#include "glad_glfw3.h"
#include "ScreenQuad.hpp"

namespace spark
{
namespace resources
{
    class Shader;
}

class Bloom
{
    public:
    Bloom() = default;
    Bloom(const Bloom&) = delete;
    Bloom(Bloom&&) = delete;
    Bloom& operator=(const Bloom&) = delete;
    Bloom& operator=(Bloom&&) = delete;
    ~Bloom();

    void setup(unsigned int width, unsigned int height);
    GLuint process(const ScreenQuad& screenQuad, GLuint lightingTexture, GLuint brightPassTexture);
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);
    void cleanup();

    float intensity = 0.2f;
    float threshold = 0.5f;
    float thresholdSize = 1.0f;

    private:
    void downsampleTexture(const ScreenQuad& screenQuad, GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight,
                           bool downscale = true);
    void upsampleTexture(const ScreenQuad& screenQuad, GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight,
                         float bloomIntensity = 1.0f);
    unsigned int w{}, h{};

    GLuint fboMip0{};
    GLuint fboMip1{}, textureMip1{};
    GLuint fboMip2{}, textureMip2{};
    GLuint fboMip3{}, textureMip3{};
    GLuint fboMip4{}, textureMip4{};

    std::shared_ptr<resources::Shader> bloomDownScaleShaderMip0ToMip1{nullptr};
    std::shared_ptr<resources::Shader> bloomDownScaleShader{nullptr};
    std::shared_ptr<resources::Shader> bloomUpsamplerShader{nullptr};
};
}  // namespace spark