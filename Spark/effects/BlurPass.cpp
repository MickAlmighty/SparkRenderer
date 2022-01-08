#include "BlurPass.h"

#include <memory>

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
void BlurPass::blurTexture(GLuint texture) const
{
    PUSH_DEBUG_GROUP(GAUSSIAN_BLUR);
    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, vFramebuffer);
    utils::bindTexture2D(vFramebuffer, vTexture);

    gaussianBlurShader->use();
    gaussianBlurShader->setVec2("inverseScreenSize", {1.0f / static_cast<float>(w), 1.0f / static_cast<float>(h)});
    gaussianBlurShader->setBool("horizontal", false);
    glBindTextureUnit(0, texture);
    screenQuad.draw();

    utils::bindTexture2D(vFramebuffer, hTexture);
    gaussianBlurShader->setBool("horizontal", true);
    glBindTextureUnit(0, vTexture);

    screenQuad.draw();
    glBindTextures(0, 1, nullptr);
    glViewport(0, 0, Spark::get().getRenderingContext().width, Spark::get().getRenderingContext().height);

    POP_DEBUG_GROUP();
}

GLuint BlurPass::getBlurredTexture() const
{
    return hTexture;
}

void BlurPass::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createGlObjects();
}

BlurPass::BlurPass(unsigned int width_, unsigned int height_) : w(width_), h(height_)
{
    gaussianBlurShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("gaussianBlur.glsl");
    createGlObjects();
}

BlurPass::~BlurPass()
{
    GLuint textures[] = {vTexture, hTexture};
    glDeleteTextures(2, textures);
    vTexture = hTexture = 0;

    GLuint framebuffers[] = {vFramebuffer};
    glDeleteFramebuffers(1, framebuffers);
    vFramebuffer = 0;
}

void BlurPass::createGlObjects()
{
    utils::recreateTexture2D(hTexture, w, h, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(vTexture, w, h, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(vFramebuffer, {vTexture});
}
}  // namespace spark::effects
