#include "BloomPass.hpp"

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
BloomPass::~BloomPass()
{
    cleanup();
}

void BloomPass::setup(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    bloomDownScaleShaderMip0ToMip1 = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("bloomDownScaleMip0ToMip1.glsl");
    bloomDownScaleShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("bloomDownScale.glsl");
    bloomUpsamplerShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("bloomUpsampler.glsl");
}

GLuint BloomPass::process(GLuint lightingTexture, GLuint brightPassTexture)
{
    PUSH_DEBUG_GROUP(BLOOM)
    downsampleFromMip0ToMip1(brightPassTexture);

    PUSH_DEBUG_GROUP(DOWNSAMPLE_FROM_MIP_1_TO_MIP_5)
    downsampleTexture(fboMip2, textureMip1, w / 4, h / 4);
    downsampleTexture(fboMip3, textureMip2, w / 8, h / 8);
    downsampleTexture(fboMip4, textureMip3, w / 16, h / 16);
    downsampleTexture(fboMip5, textureMip4, w / 32, h / 32);
    POP_DEBUG_GROUP();

    PUSH_DEBUG_GROUP(UPSAMPLE)
    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    glEnable(GL_BLEND);

    upsampleTexture(fboMip4, textureMip5, w / 16, h / 16, radiusMip4);
    upsampleTexture(fboMip3, textureMip4, w / 8, h / 8, radiusMip3);
    upsampleTexture(fboMip2, textureMip3, w / 4, h / 4, radiusMip2);
    upsampleTexture(fboMip1, textureMip2, w / 2, h / 2, radiusMip1);
    utils::bindTexture2D(fboMip0, lightingTexture);
    upsampleTexture(fboMip0, textureMip1, w, h, radiusMip0, intensity);

    glDisable(GL_BLEND);

    glViewport(0, 0, w, h);
    POP_DEBUG_GROUP();
    POP_DEBUG_GROUP();

    return lightingTexture;
}

void BloomPass::downsampleFromMip0ToMip1(GLuint brightPassTexture)
{
    PUSH_DEBUG_GROUP(DOWNSAMPLE_MIP_0_TO_MIP_1)
    glViewport(0, 0, w / 2, h / 2);
    glBindFramebuffer(GL_FRAMEBUFFER, fboMip1);

    bloomDownScaleShaderMip0ToMip1->use();
    bloomDownScaleShaderMip0ToMip1->setVec2("outputTextureSizeInversion", glm::vec2(1.0f / (w / 2.0f), 1.0f / (w / 2.0f)));
    bloomDownScaleShaderMip0ToMip1->setFloat("threshold", threshold);
    bloomDownScaleShaderMip0ToMip1->setFloat("thresholdSize", thresholdSize);
    glBindTextureUnit(0, brightPassTexture);
    screenQuad.draw();
    POP_DEBUG_GROUP()
}

void BloomPass::downsampleTexture(GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight)
{
    glViewport(0, 0, viewportWidth, viewportHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    bloomDownScaleShader->use();
    const auto lowerMipTextureSize = glm::vec2(1.0f / (viewportWidth / 2.0f), 1.0f / (viewportWidth / 2.0f));
    bloomDownScaleShader->setVec2("outputTextureSizeInversion", lowerMipTextureSize);
    glBindTextureUnit(0, texture);
    screenQuad.draw();
}

void BloomPass::upsampleTexture(GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight, float radius, float bloomIntensity)
{
    glViewport(0, 0, viewportWidth, viewportHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    bloomUpsamplerShader->use();
    bloomUpsamplerShader->setFloat("intensity", bloomIntensity);
    bloomUpsamplerShader->setFloat("radius", radius);
    bloomUpsamplerShader->setVec2("outputTextureSizeInversion", glm::vec2(1.0f / viewportWidth, 1.0f / viewportWidth));
    glBindTextureUnit(0, texture);
    screenQuad.draw();
};

void BloomPass::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    utils::recreateTexture2D(textureMip1, w / 2, h / 2, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(textureMip2, w / 4, h / 4, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(textureMip3, w / 8, h / 8, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(textureMip4, w / 16, h / 16, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(textureMip5, w / 32, h / 32, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(fboMip1, {textureMip1});
    utils::recreateFramebuffer(fboMip2, {textureMip2});
    utils::recreateFramebuffer(fboMip3, {textureMip3});
    utils::recreateFramebuffer(fboMip4, {textureMip4});
    utils::recreateFramebuffer(fboMip5, {textureMip5});
    utils::recreateFramebuffer(fboMip0, {});
}

void BloomPass::cleanup()
{
    GLuint textures[5] = {textureMip1, textureMip2, textureMip3, textureMip4, textureMip5};
    glDeleteTextures(5, textures);

    GLuint framebuffers[6] = {fboMip1, fboMip2, fboMip3, fboMip4, fboMip5, fboMip0};
    glDeleteFramebuffers(6, framebuffers);
}
}  // namespace spark::effects