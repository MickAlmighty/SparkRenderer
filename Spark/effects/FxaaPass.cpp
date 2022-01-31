#include "FxaaPass.hpp"

#include "utils/CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
FxaaPass::FxaaPass(unsigned width, unsigned height) : w(width), h(height)
{
    fxaaShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("fxaa.glsl");
    createFrameBuffersAndTextures();
}

FxaaPass::~FxaaPass()
{
    glDeleteFramebuffers(1, &fxaaFramebuffer);
}

GLuint FxaaPass::process(GLuint inputTexture)
{
    PUSH_DEBUG_GROUP(FXAA);

    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, fxaaFramebuffer);

    fxaaShader->use();

    glBindTextureUnit(0, inputTexture);
    screenQuad.draw();
    glBindTextureUnit(0, 0);

    POP_DEBUG_GROUP();

    return fxaaTexture.get();
}

void FxaaPass::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createFrameBuffersAndTextures();
}

void FxaaPass::createFrameBuffersAndTextures()
{
    fxaaTexture = utils::createTexture2D(w, h, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(fxaaFramebuffer, {fxaaTexture.get()});
}
}  // namespace spark::effects