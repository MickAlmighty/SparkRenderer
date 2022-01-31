#pragma once

#include <array>

#include "Buffer.hpp"
#include "Shader.h"
#include "ScreenQuad.hpp"
#include "utils/GlHandle.hpp"

namespace spark
{
class ICamera;
}

namespace spark::effects
{
class AmbientOcclusion
{
    public:
    AmbientOcclusion(unsigned int width, unsigned int height);
    AmbientOcclusion(const AmbientOcclusion&) = delete;
    AmbientOcclusion(AmbientOcclusion&&) = delete;
    AmbientOcclusion& operator=(const AmbientOcclusion&) = delete;
    AmbientOcclusion& operator=(AmbientOcclusion&&) = delete;
    ~AmbientOcclusion();

    GLuint process(GLuint depthTexture, GLuint normalsTexture, const std::shared_ptr<ICamera>& camera);
    void resize(unsigned int width, unsigned int height);

    int kernelSize = 32;
    float radius = 0.7f;
    float bias = 0.035f;
    float power = 4.0f;

    private:
    void createFrameBuffersAndTextures();

    static std::array<glm::vec4, 64> generateSsaoSamples();
    static std::array<glm::vec3, 16> generateSsaoNoise();

    unsigned int w{}, h{};
    GLuint ssaoFramebuffer{};
    utils::TextureHandle ssaoTexture{}, ssaoTexture2{}, randomNormalsTexture{};
    UniformBuffer samplesUbo{};
    ScreenQuad screenQuad{};
    std::shared_ptr<resources::Shader> ssaoShader{nullptr};
    std::shared_ptr<resources::Shader> ssaoBlurShader{nullptr};
    std::shared_ptr<resources::Shader> colorInversionShader{nullptr};
};
}  // namespace spark::effects