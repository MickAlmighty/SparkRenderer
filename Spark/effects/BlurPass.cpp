#include "BlurPass.h"

#include <memory>

#include "utils/CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
void BlurPass::blurTexture(GLuint texture) const
{
    PUSH_DEBUG_GROUP(GAUSSIAN_BLUR);
    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, vFramebuffer);
    utils::bindTexture2D(vFramebuffer, vTexture.get());

    horizontalGaussianBlurShader->use();
    horizontalGaussianBlurShader->setVec2("u_Uniforms.inverseScreenSize", {1.0f / static_cast<float>(w), 1.0f / static_cast<float>(h)});
    glBindTextureUnit(0, texture);
    screenQuad.draw();

    utils::bindTexture2D(vFramebuffer, hTexture.get());
    verticalGaussianBlurShader->use();
    verticalGaussianBlurShader->setVec2("u_Uniforms.inverseScreenSize", {1.0f / static_cast<float>(w), 1.0f / static_cast<float>(h)});
    glBindTextureUnit(0, vTexture.get());

    screenQuad.draw();
    glBindTextures(0, 1, nullptr);

    POP_DEBUG_GROUP();
}

GLuint BlurPass::getBlurredTexture() const
{
    return hTexture.get();
}

void BlurPass::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createGlObjects();
}

BlurPass::BlurPass(unsigned int width_, unsigned int height_) : w(width_), h(height_)
{
    horizontalGaussianBlurShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/horizontalGaussianBlur.glsl");
    verticalGaussianBlurShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/verticalGaussianBlur.glsl");
    createGlObjects();
}

BlurPass::~BlurPass()
{
    GLuint framebuffers[] = {vFramebuffer};
    glDeleteFramebuffers(1, framebuffers);
    vFramebuffer = 0;
}

void BlurPass::createGlObjects()
{
    hTexture = utils::createTexture2D(w, h, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    vTexture = utils::createTexture2D(w, h, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(vFramebuffer, {vTexture.get()});
}
}  // namespace spark::effects
