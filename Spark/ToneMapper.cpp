#include "ToneMapper.hpp"

#include "Clock.h"
#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"
#include "glad_glfw3.h"

namespace spark
{
ToneMapper::~ToneMapper()
{
    cleanup();
}

void ToneMapper::setup(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    toneMappingShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("toneMapping.glsl");
    luminanceHistogramComputeShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("luminanceHistogramCompute.glsl");
    averageLuminanceComputeShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("averageLuminanceCompute.glsl");

    luminanceHistogramComputeShader->bindSSBO("LuminanceHistogram", luminanceHistogram);
    averageLuminanceComputeShader->bindSSBO("LuminanceHistogram", luminanceHistogram);

    luminanceHistogram.resizeBuffer(256 * sizeof(uint32_t));
    utils::recreateTexture2D(averageLuminanceTexture, 1, 1, GL_R16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
    screenQuad.setup();
}

GLuint ToneMapper::process(GLuint colorTexture)
{
    PUSH_DEBUG_GROUP(TONE_MAPPING);

    calculateAverageLuminance(colorTexture);

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

void ToneMapper::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    utils::recreateTexture2D(toneMappingTexture, width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(toneMappingFramebuffer, {toneMappingTexture});
}

void ToneMapper::cleanup()
{
    glDeleteTextures(1, &toneMappingTexture);
    glDeleteTextures(1, &averageLuminanceTexture);
    glDeleteFramebuffers(1, &toneMappingFramebuffer);
    toneMappingFramebuffer = averageLuminanceTexture = toneMappingTexture = 0;
}

void ToneMapper::calculateAverageLuminance(GLuint colorTexture)
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
}  // namespace spark