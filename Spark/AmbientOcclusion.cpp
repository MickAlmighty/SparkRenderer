#include "AmbientOcclusion.hpp"

#include <random>

#include <glm/gtx/compatibility.hpp>

#include "CommonUtils.h"
#include "Spark.h"

namespace spark
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

    unsigned char red = 255;
    utils::createTexture2D(ssaoDisabledTexture, 1, 1, GL_RED, GL_RED, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_NEAREST, false, &red);

    ssaoShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("ssao.glsl");
    colorInversionShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("colorInversion.glsl");
    ssaoShader->bindUniformBuffer("Samples", samplesUbo);
    ssaoShader->bindUniformBuffer("Camera", cameraUbo);

    ssaoBlurPass = std::make_unique<BlurPass>(w, h);
    screenQuad.setup();
}

void AmbientOcclusion::cleanup()
{
    glDeleteTextures(1, &randomNormalsTexture);
    glDeleteTextures(1, &ssaoDisabledTexture);
    glDeleteTextures(1, &ssaoTexture);
    glDeleteFramebuffers(1, &ssaoFramebuffer);
}

GLuint AmbientOcclusion::process(GLuint depthTexture, GLuint normalsTexture)
{
    PUSH_DEBUG_GROUP(SSAO);

    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, ssaoFramebuffer);

    GLuint textures[3] = { depthTexture, normalsTexture, randomNormalsTexture};
    glBindTextures(0, 3, textures);
    ssaoShader->use();
    ssaoShader->setInt("kernelSize", kernelSize);
    ssaoShader->setFloat("radius", radius);
    ssaoShader->setFloat("bias", bias);
    ssaoShader->setFloat("power", power);
    ssaoShader->setVec2("screenSize", {static_cast<float>(w), static_cast<float>(h)});
    // uniforms have default values in shader
    screenQuad.draw();
    glBindTextures(0, 3, nullptr);

    ssaoBlurPass->blurTexture(ssaoTexture);

    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, ssaoFramebuffer);
    glBindTextureUnit(0, ssaoBlurPass->getBlurredTexture());
    colorInversionShader->use();
    screenQuad.draw();

    POP_DEBUG_GROUP();

    return ssaoTexture;
}

void AmbientOcclusion::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    ssaoBlurPass->recreateWithNewSize(w, h);
    utils::recreateTexture2D(ssaoTexture, w, h, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(ssaoFramebuffer, {ssaoTexture});
}

std::array<glm::vec4, 64> AmbientOcclusion::generateSsaoSamples()
{
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
    std::default_random_engine generator{};
    std::array<glm::vec4, 64> ssaoKernel;
    for (unsigned int i = 0; i < 64; ++i)
    {
        glm::vec4 sample(randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator), 0.0f);
        sample = glm::normalize(sample);
        sample *= randomFloats(generator);

        float scale = static_cast<float>(i) / 64.0f;
        scale = glm::lerp(0.1f, 1.0f, scale * scale);
        sample *= scale;

        ssaoKernel[i] = sample;
    }

    return ssaoKernel;
}

std::array<glm::vec3, 16> AmbientOcclusion::generateSsaoNoise()
{
    std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
    std::default_random_engine generator{};

    std::array<glm::vec3, 16> ssaoNoise;
    for (unsigned int i = 0; i < 16; i++)
    {
        glm::vec3 noise(randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator) * 2.0f - 1.0f, 0.0f);
        ssaoNoise[i] = noise;
    }

    return ssaoNoise;
}
}  // namespace spark