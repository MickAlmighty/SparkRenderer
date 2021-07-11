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

class BlurPass;

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

    float intensity = 0.1f;

    private:
    void downsampleTexture(const ScreenQuad& screenQuad, GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight,
                           bool downscale = true);
    void upsampleTexture(const ScreenQuad& screenQuad, GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight,
                         float bloomIntensity = 1.0f);
    unsigned int w{}, h{};

    GLuint bloomFramebuffer{}, bloomTexture{};
    GLuint downsampleFramebuffer2{}, downsampleTexture2{};
    GLuint downsampleFramebuffer4{}, downsampleTexture4{};
    GLuint downsampleFramebuffer8{}, downsampleTexture8{};
    GLuint downsampleFramebuffer16{}, downsampleTexture16{};

    std::unique_ptr<BlurPass> upsampleBloomBlurPass2;
    std::unique_ptr<BlurPass> upsampleBloomBlurPass4;
    std::unique_ptr<BlurPass> upsampleBloomBlurPass8;
    std::unique_ptr<BlurPass> upsampleBloomBlurPass16;

    std::shared_ptr<resources::Shader> bloomDownScaleShader{nullptr};
    std::shared_ptr<resources::Shader> bloomUpScaleShader{nullptr};
};
}  // namespace spark