#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#define PUSH_DEBUG_GROUP(x)                                                                                       \
    {                                                                                                             \
        std::string message = #x;                                                                                 \
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, static_cast<GLsizei>(message.length()), message.data()); \
    }

#define POP_DEBUG_GROUP() glPopDebugGroup()

namespace spark
{
namespace utils
{
    void createTexture(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping,
                       GLenum textureSampling, bool mipMaps = false);

    void bindDepthTexture(GLuint& framebuffer, GLuint depthTexture);
    void createFramebuffer(GLuint& framebuffer, const std::vector<GLuint>&& colorTextures, GLuint renderbuffer = 0);
}  // namespace utils
}  // namespace spark

#endif