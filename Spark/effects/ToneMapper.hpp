#pragma once

#include <memory>

#include "Buffer.hpp"
#include "ScreenQuad.hpp"
#include "TexturePass.hpp"

namespace spark::resources
{
class Shader;
}

namespace spark::effects
{
class ToneMapper
{
    public:
    ToneMapper(unsigned int width, unsigned int height);
    ToneMapper(const ToneMapper&) = delete;
    ToneMapper(ToneMapper&&) = delete;
    ToneMapper& operator=(const ToneMapper&) = delete;
    ToneMapper& operator=(ToneMapper&&) = delete;
    ~ToneMapper();

    GLuint process(GLuint inputTexture);
    void resize(unsigned int width, unsigned int height);

    float minLogLuminance = 0.5f;
    float logLuminanceRange = 12.0f;
    float tau = 1.1f;

    private:
    void calculateAverageLuminance();
    void createFrameBuffersAndTextures();

    unsigned int w{}, h{};
    GLuint toneMappingFramebuffer{}, toneMappingTexture{}, averageLuminanceTexture{}, colorTexture{};
    ScreenQuad screenQuad{};
    SSBO luminanceHistogram{};
    TexturePass texturePass{};
    std::shared_ptr<resources::Shader> toneMappingShader{nullptr};
    std::shared_ptr<resources::Shader> luminanceHistogramComputeShader{nullptr};
    std::shared_ptr<resources::Shader> averageLuminanceComputeShader{nullptr};
};
}  // namespace spark::effects