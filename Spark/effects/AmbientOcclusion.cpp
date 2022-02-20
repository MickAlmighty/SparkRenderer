#include "AmbientOcclusion.hpp"

#include <random>

#include <glm/gtx/compatibility.hpp>

#include "ICamera.hpp"
#include "utils/CommonUtils.h"
#include "Spark.h"

namespace spark::effects
{
AmbientOcclusion::AmbientOcclusion(unsigned int width, unsigned int height) : w(width), h(height)
{
    const auto ssaoKernel = generateSsaoSamples();
    samplesUbo.resizeBuffer(ssaoKernel.size() * sizeof(glm::vec4));
    samplesUbo.updateData(ssaoKernel);

    auto ssaoNoise = generateSsaoNoise();
    randomNormalsTexture = utils::createTexture2D(4, 4, GL_RGB32F, GL_RGB, GL_FLOAT, GL_REPEAT, GL_NEAREST, false, ssaoNoise.data());

    ssaoShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/ssao.glsl");
    ssaoBlurShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/ssaoBlur.glsl");
    colorInversionShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/colorInversion.glsl");
    ssaoShader->bindUniformBuffer("Samples", samplesUbo);
    createFrameBuffersAndTextures();
}

AmbientOcclusion::~AmbientOcclusion()
{
    glDeleteFramebuffers(1, &ssaoFramebuffer);
}

GLuint AmbientOcclusion::process(GLuint depthTexture, GLuint normalsTexture, const std::shared_ptr<ICamera>& camera)
{
    PUSH_DEBUG_GROUP(SSAO);

    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, ssaoFramebuffer);
    glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint textures[3] = {depthTexture, normalsTexture, randomNormalsTexture.get()};
    glBindTextures(0, 3, textures);
    ssaoShader->use();
    ssaoShader->setInt("u_Uniforms.kernelSize", kernelSize);
    ssaoShader->setFloat("u_Uniforms.radius", radius);
    ssaoShader->setFloat("u_Uniforms.bias", bias);
    ssaoShader->setFloat("u_Uniforms.power", power);
    ssaoShader->setVec2("u_Uniforms.screenSize", {static_cast<float>(w), static_cast<float>(h)});
    ssaoShader->bindUniformBuffer("Camera", camera->getUbo());

    screenQuad.draw();
    glBindTextures(0, 3, nullptr);

    utils::bindTexture2D(ssaoFramebuffer, ssaoTexture2.get());
    ssaoBlurShader->use();
    ssaoBlurShader->setInt("u_Uniforms.blurSize", 4);
    glBindTextureUnit(0, ssaoTexture.get());
    screenQuad.draw();

    utils::bindTexture2D(ssaoFramebuffer, ssaoTexture.get());
    glBindTextureUnit(0, ssaoTexture2.get());
    colorInversionShader->use();
    screenQuad.draw();

    POP_DEBUG_GROUP();

    return ssaoTexture.get();
}

void AmbientOcclusion::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createFrameBuffersAndTextures();
}

void AmbientOcclusion::createFrameBuffersAndTextures()
{
    ssaoTexture = utils::createTexture2D(w, h, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    ssaoTexture2 = utils::createTexture2D(w, h, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(ssaoFramebuffer, {ssaoTexture.get()});
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