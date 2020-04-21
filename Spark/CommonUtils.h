#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

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
    void createTexture2D(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat,
                         GLenum textureWrapping, GLenum textureSampling, bool mipMaps = false, void* data = nullptr);
    void createCubemap(GLuint& texture, unsigned int size, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping,
                       GLenum textureSampling, bool mipMaps = false);

    void bindDepthTexture(GLuint& framebuffer, GLuint depthTexture);
    void createFramebuffer(GLuint& framebuffer, const std::vector<GLuint>&& colorTextures, GLuint renderbuffer = 0);

    GLuint createBrdfLookupTexture(unsigned int size);

    template<typename T>
    void uploadDataToTexture2D(GLuint texture, GLuint mipMapLevel, GLuint width, GLuint height, GLenum format, GLenum type,
                               const std::vector<T>& buffer)
    {
        glTextureSubImage2D(texture, mipMapLevel, 0, 0, width, height, format, type, buffer.data());
    }

    glm::mat4 getProjectionReversedZInfFar(uint32_t width, uint32_t height, float fovDegrees, float zNear);
    glm::mat4 getProjectionReversedZ(uint32_t width, uint32_t height, float fovDegrees, float zNear, float zFar);
}  // namespace utils
}  // namespace spark

#endif