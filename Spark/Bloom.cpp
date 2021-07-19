#include "Bloom.hpp"

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark
{
Bloom::~Bloom()
{
    cleanup();
}

void Bloom::setup(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    bloomDownScaleShaderMip0ToMip1 = Spark::resourceLibrary.getResourceByName<resources::Shader>("bloomDownScaleMip0ToMip1.glsl");
    bloomDownScaleShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("bloomDownScale.glsl");
    bloomUpsamplerShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("bloomUpsampler.glsl");
}

GLuint Bloom::process(const ScreenQuad& screenQuad, GLuint lightingTexture, GLuint brightPassTexture)
{
    PUSH_DEBUG_GROUP(BLOOM)
    glViewport(0, 0, w / 2, h / 2);
    glBindFramebuffer(GL_FRAMEBUFFER, fboMip1);

    bloomDownScaleShaderMip0ToMip1->use();
    bloomDownScaleShaderMip0ToMip1->setVec2("outputTextureSizeInversion", glm::vec2(1.0f / (w / 2.0f), 1.0f / (w / 2.0f)));
    bloomDownScaleShaderMip0ToMip1->setFloat("threshold", threshold);
    bloomDownScaleShaderMip0ToMip1->setFloat("thresholdSize", thresholdSize);
    glBindTextureUnit(0, brightPassTexture);
    screenQuad.draw();

    downsampleTexture(screenQuad, fboMip2, textureMip1, w / 4, h / 4);
    downsampleTexture(screenQuad, fboMip3, textureMip2, w / 8, h / 8);
    downsampleTexture(screenQuad, fboMip4, textureMip3, w / 16, h / 16);

    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    glEnable(GL_BLEND);

    upsampleTexture(screenQuad, fboMip3, textureMip4, w / 8, h / 8);
    upsampleTexture(screenQuad, fboMip2, textureMip3, w / 4, h / 4);
    upsampleTexture(screenQuad, fboMip1, textureMip2, w / 2, h / 2);
    utils::bindTexture2D(fboMip0, lightingTexture);
    upsampleTexture(screenQuad, fboMip0, textureMip1, w, h, intensity);

    glDisable(GL_BLEND);

    glViewport(0, 0, w, h);
    POP_DEBUG_GROUP();

    return lightingTexture;
}

void Bloom::downsampleTexture(const ScreenQuad& screenQuad, GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight,
                              bool downscale)
{
    glViewport(0, 0, viewportWidth, viewportHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    bloomDownScaleShader->use();
    const auto lowerMipTextureSize = glm::vec2(1.0f / (viewportWidth / 2.0f), 1.0f / (viewportWidth / 2.0f));
    bloomDownScaleShader->setVec2("outputTextureSizeInversion", lowerMipTextureSize);
    glBindTextureUnit(0, texture);
    screenQuad.draw();
}

void Bloom::upsampleTexture(const ScreenQuad& screenQuad, GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight,
                            float bloomIntensity)
{
    glViewport(0, 0, viewportWidth, viewportHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    bloomUpsamplerShader->use();
    bloomUpsamplerShader->setFloat("intensity", bloomIntensity);
    bloomUpsamplerShader->setVec2("outputTextureSizeInversion", glm::vec2(1.0f / viewportWidth,1.0f / viewportWidth));
    glBindTextureUnit(0, texture);
    screenQuad.draw();
};

void Bloom::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    utils::recreateTexture2D(textureMip1, w / 2, h / 2, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(textureMip2, w / 4, h / 4, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(textureMip3, w / 8, h / 8, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(textureMip4, w / 16, h / 16, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(fboMip1, {textureMip1});
    utils::recreateFramebuffer(fboMip2, {textureMip2});
    utils::recreateFramebuffer(fboMip3, {textureMip3});
    utils::recreateFramebuffer(fboMip4, {textureMip4});
    utils::recreateFramebuffer(fboMip0, {});
}

void Bloom::cleanup()
{
    GLuint textures[4] = {textureMip1, textureMip2, textureMip3, textureMip4};
    glDeleteTextures(4, textures);

    GLuint framebuffers[5] = {fboMip1, fboMip2, fboMip3, fboMip4, fboMip0};
    glDeleteFramebuffers(5, framebuffers);
}
}  // namespace spark