#pragma once

#include <array>

#include "BlurPass.h"
#include "Buffer.hpp"
#include "GBuffer.h"
#include "Shader.h"
#include "Structs.h"

namespace spark
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

    GLuint ssaoFramebuffer{}, ssaoTexture{}, randomNormalsTexture{}, ssaoDisabledTexture{};
    UniformBuffer samplesUbo{};
    ScreenQuad screenQuad{};
    std::shared_ptr<resources::Shader> ssaoShader{nullptr};
    std::shared_ptr<resources::Shader> colorInversionShader{nullptr};
    std::unique_ptr<BlurPass> ssaoBlurPass;

    unsigned int w{}, h{};
};
}  // namespace spark