#pragma once

#include <array>

#include "Buffer.hpp"
#include "Shader.h"
#include "Structs.h"
#include "ScreenQuad.hpp"

namespace spark::effects
{
class AmbientOcclusion
{
    public:
    AmbientOcclusion() = default;
    AmbientOcclusion(const AmbientOcclusion&) = delete;
    AmbientOcclusion(AmbientOcclusion&&) = delete;
    AmbientOcclusion& operator=(const AmbientOcclusion&) = delete;
    AmbientOcclusion& operator=(AmbientOcclusion&&) = delete;
    ~AmbientOcclusion();

    void setup(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo);
    void cleanup();
    GLuint process(GLuint depthTexture, GLuint normalsTexture);
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);

    int kernelSize = 32;
    float radius = 0.7f;
    float bias = 0.035f;
    float power = 4.0f;

    private:
    static std::array<glm::vec4, 64> generateSsaoSamples();
    static std::array<glm::vec3, 16> generateSsaoNoise();

    GLuint ssaoFramebuffer{}, ssaoTexture{}, ssaoTexture2{}, randomNormalsTexture{};
    UniformBuffer samplesUbo{};
    ScreenQuad screenQuad{};
    std::shared_ptr<resources::Shader> ssaoShader{nullptr};
    std::shared_ptr<resources::Shader> ssaoBlurShader{nullptr};
    std::shared_ptr<resources::Shader> colorInversionShader{nullptr};

    unsigned int w{}, h{};
};
}  // namespace spark::effects