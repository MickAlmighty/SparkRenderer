#include "Bloom.hpp"

#include "BlurPass.h"
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
    bloomDownScaleShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("bloomDownScale.glsl");
    bloomUpScaleShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("bloomUpScale.glsl");

    upsampleBloomBlurPass2 = std::make_unique<BlurPass>(width / 2, height / 2);
    upsampleBloomBlurPass4 = std::make_unique<BlurPass>(width / 4, height / 4);
    upsampleBloomBlurPass8 = std::make_unique<BlurPass>(width / 8, height / 8);
    upsampleBloomBlurPass16 = std::make_unique<BlurPass>(width / 16, height / 16);
}

GLuint Bloom::process(const ScreenQuad& screenQuad, GLuint lightingTexture, GLuint brightPassTexture)
{
    PUSH_DEBUG_GROUP(BLOOM)
    upsampleTexture(screenQuad, bloomFramebuffer, lightingTexture, w, h);

    downsampleTexture(screenQuad, downsampleFramebuffer2, brightPassTexture, w / 2, h / 2);
    downsampleTexture(screenQuad, downsampleFramebuffer4, downsampleTexture2, w / 4, h / 4);
    downsampleTexture(screenQuad, downsampleFramebuffer8, downsampleTexture4, w / 8, h / 8);
    downsampleTexture(screenQuad, downsampleFramebuffer16, downsampleTexture8, w / 16, h / 16);

    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    glEnable(GL_BLEND);

    upsampleBloomBlurPass16->blurTexture(downsampleTexture16);
    upsampleTexture(screenQuad, downsampleFramebuffer8, upsampleBloomBlurPass16->getBlurredTexture(), w / 8, h / 8);

    upsampleBloomBlurPass8->blurTexture(downsampleTexture8);
    upsampleTexture(screenQuad, downsampleFramebuffer4, upsampleBloomBlurPass8->getBlurredTexture(), w / 4, h / 4);

    upsampleBloomBlurPass4->blurTexture(downsampleTexture4);
    upsampleTexture(screenQuad, downsampleFramebuffer2, upsampleBloomBlurPass4->getBlurredTexture(), w / 2, h / 2);

    upsampleBloomBlurPass2->blurTexture(downsampleTexture2);
    upsampleTexture(screenQuad, bloomFramebuffer, upsampleBloomBlurPass2->getBlurredTexture(), w, h, intensity);

    glDisable(GL_BLEND);

    glViewport(0, 0, w, h);
    POP_DEBUG_GROUP();

    return bloomTexture;
}

void Bloom::downsampleTexture(const ScreenQuad& screenQuad, GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight,
                              bool downscale)
{
    glViewport(0, 0, viewportWidth, viewportHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    bloomDownScaleShader->use();
    bloomDownScaleShader->setVec2("outputTextureSizeInversion",
                                  glm::vec2(1.0f / static_cast<float>(viewportWidth), 1.0f / static_cast<float>(viewportHeight)));
    glBindTextureUnit(0, texture);
    screenQuad.draw();
}

void Bloom::upsampleTexture(const ScreenQuad& screenQuad, GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight,
                            float bloomIntensity)
{
    glViewport(0, 0, viewportWidth, viewportHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    bloomUpScaleShader->use();
    bloomUpScaleShader->setFloat("intensity", bloomIntensity);
    glBindTextureUnit(0, texture);
    screenQuad.draw();
};

void Bloom::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    utils::recreateTexture2D(downsampleTexture2, width / 2, height / 2, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(downsampleTexture4, width / 4, height / 4, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(downsampleTexture8, width / 8, height / 8, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(downsampleTexture16, width / 16, height / 16, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(bloomTexture, width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(downsampleFramebuffer2, {downsampleTexture2});
    utils::recreateFramebuffer(downsampleFramebuffer4, {downsampleTexture4});
    utils::recreateFramebuffer(downsampleFramebuffer8, {downsampleTexture8});
    utils::recreateFramebuffer(downsampleFramebuffer16, {downsampleTexture16});
    utils::recreateFramebuffer(bloomFramebuffer, {bloomTexture});
}

void Bloom::cleanup()
{
    GLuint textures[5] = {downsampleTexture2, downsampleTexture4, downsampleTexture8, downsampleTexture16, bloomTexture};
    glDeleteTextures(5, textures);

    GLuint framebuffers[5] = {downsampleFramebuffer2, downsampleFramebuffer4, downsampleFramebuffer8, downsampleFramebuffer16, bloomFramebuffer};
    glDeleteFramebuffers(5, framebuffers);
}
}  // namespace spark