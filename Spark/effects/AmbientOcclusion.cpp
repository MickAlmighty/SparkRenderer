#include "AmbientOcclusion.hpp"

#include <random>

#include <glm/gtx/compatibility.hpp>

#include "CommonUtils.h"
#include "Spark.h"

namespace spark::effects
{
AmbientOcclusion::~AmbientOcclusion()
{
    cleanup();
}

void AmbientOcclusion::setup(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo)
{
    w = width;
    h = height;
    const auto ssaoKernel = generateSsaoSamples();
    samplesUbo.resizeBuffer(ssaoKernel.size() * sizeof(glm::vec4));
    samplesUbo.updateData(ssaoKernel);

    auto ssaoNoise = generateSsaoNoise();
    utils::createTexture2D(randomNormalsTexture, 4, 4, GL_RGB32F, GL_RGB, GL_FLOAT, GL_REPEAT, GL_NEAREST, false, ssaoNoise.data());

    ssaoShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("ssao.glsl");
    ssaoBlurShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("ssaoBlur.glsl");
    colorInversionShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("colorInversion.glsl");
    ssaoShader->bindUniformBuffer("Samples", samplesUbo);
    ssaoShader->bindUniformBuffer("Camera", cameraUbo);

    screenQuad.setup();
}

void AmbientOcclusion::cleanup()
{
    glDeleteTextures(1, &randomNormalsTexture);
    glDeleteTextures(1, &ssaoTexture);
    glDeleteTextures(1, &ssaoTexture2);
    glDeleteFramebuffers(1, &ssaoFramebuffer);
}

GLuint AmbientOcclusion::process(GLuint depthTexture, GLuint normalsTexture)
{
    PUSH_DEBUG_GROUP(SSAO);

    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, ssaoFramebuffer);
    glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint textures[3] = {depthTexture, normalsTexture, randomNormalsTexture};
    glBindTextures(0, 3, textures);
    ssaoShader->use();
    ssaoShader->setInt("kernelSize", kernelSize);
    ssaoShader->setFloat("radius", radius);
    ssaoShader->setFloat("bias", bias);
    ssaoShader->setFloat("power", power);
    ssaoShader->setVec2("screenSize", {static_cast<float>(w), static_cast<float>(h)});

    screenQuad.draw();
    glBindTextures(0, 3, nullptr);

    utils::bindTexture2D(ssaoFramebuffer, ssaoTexture2);
    ssaoBlurShader->use();
    glBindTextureUnit(0, ssaoTexture);
    screenQuad.draw();

    utils::bindTexture2D(ssaoFramebuffer, ssaoTexture);
    glBindTextureUnit(0, ssaoTexture2);
    colorInversionShader->use();
    screenQuad.draw();

    POP_DEBUG_GROUP();

    return ssaoTexture;
}

void AmbientOcclusion::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    utils::recreateTexture2D(ssaoTexture, w, h, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(ssaoTexture2, w, h, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(ssaoFramebuffer, {ssaoTexture});
}

std::array<glm::vec4, 64> AmbientOcclusion::generateSsaoSamples()
{
    std::uniform_real_distribution<float> randomFloats(0.0f, 1.0f);
    std::default_random_engine generator{};
    std::array<glm::vec4, 64> ssaoKernel;
    for(unsigned int i = 0; i < 64; ++i)
    {
        glm::vec3 sample(randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator));
        sample = glm::normalize(sample);
        sample *= randomFloats(generator);

        float scale = static_cast<float>(i) / 64.0f;
        scale = glm::lerp(0.1f, 1.0f, scale * scale);
        sample *= scale;

        ssaoKernel[i] = glm::vec4(sample, 0.0f);
    }

    return ssaoKernel;
}

std::array<glm::vec3, 16> AmbientOcclusion::generateSsaoNoise()
{
    std::uniform_real_distribution<float> randomFloats(0.0f, 1.0f);
    std::default_random_engine generator{};

    std::array<glm::vec3, 16> ssaoNoise;
    for(unsigned int i = 0; i < 16; i++)
    {
        glm::vec3 noise(randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator) * 2.0f - 1.0f, 0.0f);
        ssaoNoise[i] = normalize(noise);
    }

    return ssaoNoise;
}
}  // namespace spark::effects