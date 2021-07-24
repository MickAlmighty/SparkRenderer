#include "FxaaPass.hpp"

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
FxaaPass::~FxaaPass()
{
    cleanup();
}

void FxaaPass::setup(unsigned int width, unsigned int height)
{
    screenQuad.setup();
    fxaaShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("fxaa.glsl");
}

GLuint FxaaPass::process(GLuint inputTexture)
{
    PUSH_DEBUG_GROUP(FXAA);

    glBindFramebuffer(GL_FRAMEBUFFER, fxaaFramebuffer);

    fxaaShader->use();

    glBindTextureUnit(0, inputTexture);
    screenQuad.draw();
    glBindTextureUnit(0, 0);

    POP_DEBUG_GROUP();

    return fxaaTexture;
}

void FxaaPass::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    utils::recreateTexture2D(fxaaTexture, width, height, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(fxaaFramebuffer, {fxaaTexture});
}

void FxaaPass::cleanup()
{
    glDeleteTextures(1, &fxaaTexture);
    glDeleteFramebuffers(1, &fxaaFramebuffer);
}
}  // namespace spark::effects