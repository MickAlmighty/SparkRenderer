#pragma once

#include "Shader.h"

namespace spark
{
class ToneMapper
{
    public:
    ToneMapper() = default;
    ToneMapper(const ToneMapper&) = delete;
    ToneMapper(ToneMapper&&) = delete;
    ToneMapper& operator=(const ToneMapper&) = delete;
    ToneMapper& operator=(ToneMapper&&) = delete;
    ~ToneMapper();

    void setup(unsigned int width, unsigned int height);
    GLuint process(GLuint outputTexture, const ScreenQuad& screenQuad);
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);
    void cleanup();

    float minLogLuminance = 0.5f;
    float oneOverLogLuminanceRange = 1.0f / 12.0f;
    float logLuminanceRange = 12.0f;
    float tau = 1.1f;

    private:
    void calculateAverageLuminance(GLuint outputTexture);

    unsigned int w{}, h{};
    GLuint toneMappingFramebuffer{}, toneMappingTexture{}, averageLuminanceTexture{};
    SSBO luminanceHistogram{};
    std::shared_ptr<resources::Shader> toneMappingShader{nullptr};
    std::shared_ptr<resources::Shader> luminanceHistogramComputeShader{nullptr};
    std::shared_ptr<resources::Shader> averageLuminanceComputeShader{nullptr};
};
}  // namespace spark