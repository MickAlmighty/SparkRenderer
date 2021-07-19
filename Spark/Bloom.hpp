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
    
    GLuint process(GLuint lightingTexture, GLuint brightPassTexture);
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);
    void cleanup();

    float intensity = 1.0f;
    float threshold = 0.2f;
    float thresholdSize = 0.3f;
    float radiusMip0{ 4.0f };
    float radiusMip1{ 6.0f };
    float radiusMip2{ 7.5f };
    float radiusMip3{ 8.0f };
    float radiusMip4{ 9.3f };

    private:
    void downsampleFromMip0ToMip1(GLuint brightPassTexture);
    void downsampleTexture(GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight);
    void upsampleTexture(GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight, float radius, float bloomIntensity = 1.0f);

    unsigned int w{}, h{};
    ScreenQuad screenQuad{};
    GLuint fboMip0{};
    GLuint fboMip1{}, textureMip1{};
    GLuint fboMip2{}, textureMip2{};
    GLuint fboMip3{}, textureMip3{};
    GLuint fboMip4{}, textureMip4{};
    GLuint fboMip5{}, textureMip5{};

    std::shared_ptr<resources::Shader> bloomDownScaleShaderMip0ToMip1{nullptr};
    std::shared_ptr<resources::Shader> bloomDownScaleShader{nullptr};
    std::shared_ptr<resources::Shader> bloomUpsamplerShader{nullptr};
};
}  // namespace spark