#include "BlurPass.h"

#include <memory>

#include "CommonUtils.h"
#include "ResourceLibrary.h"
#include "Shader.h"
#include "Spark.h"

namespace spark
{
void BlurPass::blurTexture(GLuint texture) const
{
    PUSH_DEBUG_GROUP(GAUSSIAN_BLUR);
    glViewport(0, 0, width, height);
    glBindFramebuffer(GL_FRAMEBUFFER, vFramebuffer);
    utils::bindTexture2D(vFramebuffer, vTexture);

    gaussianBlurShader->use();
    gaussianBlurShader->setVec2("inverseScreenSize", {1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height)});
    gaussianBlurShader->setBool("horizontal", false);
    glBindTextureUnit(0, texture);
    screenQuad.draw();

    utils::bindTexture2D(vFramebuffer, hTexture);
    gaussianBlurShader->setBool("horizontal", true);
    glBindTextureUnit(0, vTexture);

    screenQuad.draw();
    glBindTextures(0, 1, nullptr);
    glViewport(0, 0, Spark::WIDTH, Spark::HEIGHT);

    POP_DEBUG_GROUP();
}

GLuint BlurPass::getBlurredTexture() const
{
    return hTexture;
}

void BlurPass::recreateWithNewSize(unsigned int width, unsigned int height)
{
    this->width = width;
    this->height = height;
    createGlObjects();
}

BlurPass::BlurPass(unsigned int width_, unsigned int height_) : width(width_), height(height_)
{
    gaussianBlurShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("gaussianBlur.glsl");
    screenQuad.setup();
    createGlObjects();
}

BlurPass::~BlurPass()
{
    deleteGlObjects();
}

void BlurPass::createGlObjects()
{
    utils::recreateTexture2D(hTexture, width, height, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(vTexture, width, height, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(vFramebuffer, {vTexture});
}

void BlurPass::deleteGlObjects()
{
    GLuint textures[] = {vTexture, hTexture};
    glDeleteTextures(2, textures);
    vTexture = hTexture = 0;

    GLuint framebuffers[] = {vFramebuffer};
    glDeleteFramebuffers(1, framebuffers);
    vFramebuffer = 0;
}
}  // namespace spark
