#include "TexturePass.hpp"

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
TexturePass::TexturePass()
{
    utils::createFramebuffer(framebuffer);
    texturePassThrough = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("texturePassThrough.glsl");
}

TexturePass::~TexturePass()
{
    glDeleteFramebuffers(1, &framebuffer);
}

void TexturePass::process(unsigned int width, unsigned int height, GLuint inputTexture, GLuint outputTexture)
{
    PUSH_DEBUG_GROUP(TEXTURE_PASS)
    glViewport(0, 0, width, height);
    utils::bindTexture2D(framebuffer, outputTexture);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    texturePassThrough->use();
    glBindTextureUnit(0, inputTexture);
    screenQuad.draw();

    POP_DEBUG_GROUP()
}
}  // namespace spark::effects