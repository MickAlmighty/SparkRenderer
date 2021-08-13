#include "ToneMapper.hpp"

#include "Clock.h"
#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"
#include "glad_glfw3.h"

namespace spark::effects
{
ToneMapper::ToneMapper(unsigned int width, unsigned int height): w(width), h(height)
{
    toneMappingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("toneMapping.glsl");
    luminanceHistogramComputeShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("luminanceHistogramCompute.glsl");
    averageLuminanceComputeShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("averageLuminanceCompute.glsl");

    luminanceHistogramComputeShader->bindSSBO("LuminanceHistogram", luminanceHistogram);
    averageLuminanceComputeShader->bindSSBO("LuminanceHistogram", luminanceHistogram);

    luminanceHistogram.resizeBuffer(256 * sizeof(uint32_t));
    utils::recreateTexture2D(averageLuminanceTexture, 1, 1, GL_R16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
    createFrameBuffersAndTextures();
}

ToneMapper::~ToneMapper()
{
    glDeleteTextures(1, &toneMappingTexture);
    glDeleteTextures(1, &averageLuminanceTexture);
    glDeleteTextures(1, &colorTexture);
    glDeleteFramebuffers(1, &toneMappingFramebuffer);
}

GLuint ToneMapper::process(GLuint inputTexture)
{
    PUSH_DEBUG_GROUP(TONE_MAPPING);

    texturePass.process(w, h, inputTexture, colorTexture);

    calculateAverageLuminance();

    glBindFramebuffer(GL_FRAMEBUFFER, toneMappingFramebuffer);

    toneMappingShader->use();
    toneMappingShader->setVec2("inversedScreenSize", {1.0f / w, 1.0f / h});

    glBindTextureUnit(0, colorTexture);
    glBindTextureUnit(1, averageLuminanceTexture);
    screenQuad.draw();
    glBindTextures(0, 3, nullptr);

    POP_DEBUG_GROUP();
    return toneMappingTexture;
}

void ToneMapper::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createFrameBuffersAndTextures();
}

void ToneMapper::createFrameBuffersAndTextures()
{
    utils::recreateTexture2D(toneMappingTexture, w, h, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(colorTexture, w, h, GL_RGBA16F, GL_RGBA, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(toneMappingFramebuffer, {toneMappingTexture});
}

void ToneMapper::calculateAverageLuminance()
{
    const float oneOverLogLuminanceRange = 1.0f / logLuminanceRange;

    // this buffer is attached to both shaders in method SparkRenderer::updateBufferBindings()
    luminanceHistogram.clearData();  // resetting histogram buffer

    // first compute dispatch
    luminanceHistogramComputeShader->use();

    luminanceHistogramComputeShader->setIVec2("inputTextureSize", glm::ivec2(w, h));
    luminanceHistogramComputeShader->setFloat("minLogLuminance", minLogLuminance);
    luminanceHistogramComputeShader->setFloat("oneOverLogLuminanceRange", oneOverLogLuminanceRange);

    glBindImageTexture(0, colorTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
    luminanceHistogramComputeShader->dispatchCompute(w / 16, h / 16, 1);  // localWorkGroups has dimensions of x = 16, y = 16
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // second compute dispatch
    averageLuminanceComputeShader->use();

    averageLuminanceComputeShader->setUInt("pixelCount", w * h);
    averageLuminanceComputeShader->setFloat("minLogLuminance", minLogLuminance);
    averageLuminanceComputeShader->setFloat("logLuminanceRange", logLuminanceRange);
    averageLuminanceComputeShader->setFloat("deltaTime", static_cast<float>(Clock::getDeltaTime()));
    averageLuminanceComputeShader->setFloat("tau", tau);

    glBindImageTexture(0, averageLuminanceTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R16F);
    averageLuminanceComputeShader->dispatchCompute(1, 1, 1);  // localWorkGroups has dimensions of x = 16, y = 16
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}
}  // namespace spark::effects