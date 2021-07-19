#pragma once

#include <memory>

#include "Buffer.hpp"
#include "ScreenQuad.hpp"

namespace spark
{
namespace resources {
    class Shader;
}

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
    GLuint process(GLuint colorTexture);
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);
    void cleanup();

    float minLogLuminance = 0.5f;
    float logLuminanceRange = 12.0f;
    float tau = 1.1f;

    private:
    void calculateAverageLuminance(GLuint colorTexture);

    unsigned int w{}, h{};
    GLuint toneMappingFramebuffer{}, toneMappingTexture{}, averageLuminanceTexture{};
    ScreenQuad screenQuad{};
    SSBO luminanceHistogram{};
    std::shared_ptr<resources::Shader> toneMappingShader{nullptr};
    std::shared_ptr<resources::Shader> luminanceHistogramComputeShader{nullptr};
    std::shared_ptr<resources::Shader> averageLuminanceComputeShader{nullptr};
};
}  // namespace spark