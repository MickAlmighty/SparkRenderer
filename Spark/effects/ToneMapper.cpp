#include "ToneMapper.hpp"

#include "Clock.h"
#include "utils/CommonUtils.h"
#include "Shader.h"
#include "Spark.h"
#include "glad_glfw3.h"

namespace spark::effects
{
ToneMapper::ToneMapper(unsigned int width, unsigned int height): w(width), h(height)
{
    toneMappingShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/toneMapping.glsl");
    luminanceHistogramComputeShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/luminanceHistogramCompute.glsl");
    averageLuminanceComputeShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/averageLuminanceCompute.glsl");

    luminanceHistogram.resizeBuffer(256 * sizeof(uint32_t));
    float initialLuminance = 1.0f;
    averageLuminanceTexture = utils::createTexture2D(1, 1, GL_R16F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST, false, reinterpret_cast<void*>(&initialLuminance));
    createFrameBuffersAndTextures();
}

ToneMapper::~ToneMapper()
{
    glDeleteFramebuffers(1, &toneMappingFramebuffer);
}

GLuint ToneMapper::process(GLuint inputTexture)
{
    PUSH_DEBUG_GROUP(TONE_MAPPING);

    texturePass.process(w, h, inputTexture, colorTexture.get());

    calculateAverageLuminance();

    glBindFramebuffer(GL_FRAMEBUFFER, toneMappingFramebuffer);

    toneMappingShader->use();
    toneMappingShader->setVec2("inversedScreenSize", {1.0f / w, 1.0f / h});

    glBindTextureUnit(0, colorTexture.get());
    glBindTextureUnit(1, averageLuminanceTexture.get());
    screenQuad.draw();
    glBindTextures(0, 3, nullptr);

    POP_DEBUG_GROUP();
    return toneMappingTexture.get();
}

void ToneMapper::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createFrameBuffersAndTextures();
}

void ToneMapper::createFrameBuffersAndTextures()
{
    toneMappingTexture = utils::createTexture2D(w, h, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    colorTexture = utils::createTexture2D(w, h, GL_RGBA16F, GL_RGBA, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(toneMappingFramebuffer, {toneMappingTexture.get()});
}

void ToneMapper::calculateAverageLuminance()
{
    const float oneOverLogLuminanceRange = 1.0f / logLuminanceRange;

    // this buffer is attached to both shaders in method SparkRenderer::updateBufferBindings()
    luminanceHistogram.clearData();  // resetting histogram buffer

    // first compute dispatch
    luminanceHistogramComputeShader->use();
    luminanceHistogramComputeShader->bindSSBO("LuminanceHistogram", luminanceHistogram);
    luminanceHistogramComputeShader->setIVec2("inputTextureSize", glm::ivec2(w, h));
    luminanceHistogramComputeShader->setFloat("minLogLuminance", minLogLuminance);
    luminanceHistogramComputeShader->setFloat("oneOverLogLuminanceRange", oneOverLogLuminanceRange);

    glBindImageTexture(0, colorTexture.get(), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
    luminanceHistogramComputeShader->dispatchCompute(w / 16, h / 16, 1);  // localWorkGroups has dimensions of x = 16, y = 16
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // second compute dispatch
    averageLuminanceComputeShader->use();
    averageLuminanceComputeShader->bindSSBO("LuminanceHistogram", luminanceHistogram);
    averageLuminanceComputeShader->setUInt("pixelCount", w * h);
    averageLuminanceComputeShader->setFloat("minLogLuminance", minLogLuminance);
    averageLuminanceComputeShader->setFloat("logLuminanceRange", logLuminanceRange);
    averageLuminanceComputeShader->setFloat("deltaTime", static_cast<float>(Clock::getDeltaTime()));
    averageLuminanceComputeShader->setFloat("tau", tau);

    glBindImageTexture(0, averageLuminanceTexture.get(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_R16F);
    averageLuminanceComputeShader->dispatchCompute(1, 1, 1);  // localWorkGroups has dimensions of x = 16, y = 16
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}
}  // namespace spark::effects