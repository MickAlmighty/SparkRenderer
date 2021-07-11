#include "GBuffer.h"

#include "CommonUtils.h"
#include "Logging.h"

namespace spark
{
void GBuffer::setup(unsigned int width, unsigned int height)
{
    utils::recreateTexture2D(colorTexture, width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_NEAREST);
    utils::recreateTexture2D(normalsTexture, width, height, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
    utils::recreateTexture2D(roughnessMetalnessTexture, width, height, GL_RG, GL_RG, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_NEAREST);
    utils::recreateTexture2D(depthTexture, width, height, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

    utils::recreateFramebuffer(framebuffer, {colorTexture, normalsTexture, roughnessMetalnessTexture});
    utils::bindDepthTexture(framebuffer, depthTexture);
}

void GBuffer::cleanup()
{
    GLuint textures[4] = {colorTexture, normalsTexture, roughnessMetalnessTexture, depthTexture};
    glDeleteTextures(4, textures);
    colorTexture = normalsTexture = roughnessMetalnessTexture = depthTexture = 0;

    glDeleteFramebuffers(1, &framebuffer);
    framebuffer = 0;
}

GBuffer::~GBuffer()
{
    if (colorTexture != 0 || framebuffer != 0)
    {
        SPARK_WARN("G-Buffer is destroyed before calling cleanup method! Cleaning resources!");
        cleanup();
    }
}
}  // namespace spark