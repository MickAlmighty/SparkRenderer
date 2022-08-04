#pragma once

#include <vector>
#include <string>

#include <glm/glm.hpp>

#include "Buffer.hpp"
#include "glad_glfw3.h"
#include "GlHandle.hpp"

#ifdef DEBUG
#    define PUSH_DEBUG_GROUP(x)                                                                               \
        {                                                                                                     \
            const char message[] = #x;                                                                        \
            glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, static_cast<GLsizei>(sizeof(message)), message); \
        }

#    define POP_DEBUG_GROUP() glPopDebugGroup();
#else
#    define PUSH_DEBUG_GROUP(x)
#    define POP_DEBUG_GROUP()
#endif

namespace spark::utils
{
[[nodiscard]] UniqueTextureHandle createCubemapArray(GLuint width, GLuint height, GLuint numberOfCubemaps, GLenum internalFormat,
                                       GLenum textureWrapping, GLenum textureSampling, uint32_t numberOfMipMaps);

[[nodiscard]] UniqueTextureHandle createTexture2D(GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat,
                                                  GLenum textureWrapping, GLenum textureSampling, bool mipMaps = false, void* data = nullptr);

[[nodiscard]] UniqueTextureHandle createCubemap(unsigned int size, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping,
                                                GLenum textureSampling, bool mipMaps = false);

void bindTexture2D(GLuint framebuffer, GLuint colorTexture, GLuint renderTargetIds = 0, GLuint mipmapLevel = 0);
void bindTextures2D(GLuint framebuffer, const std::vector<GLuint>& colorTextures, const std::vector<GLuint>& renderTargetIds = {});
void bindTextures2D(GLuint framebuffer, const std::vector<GLuint>&& colorTextures, const std::vector<GLuint>&& renderTargetIds = {});

void bindDepthTexture(GLuint& framebuffer, GLuint depthTexture);

void createFramebuffer(GLuint& framebuffer, std::vector<GLuint>&& colorTextures = {}, GLuint renderbuffer = 0);
void recreateFramebuffer(GLuint& framebuffer, std::vector<GLuint>&& colorTextures = {}, GLuint renderbuffer = 0);

TextureHandle createBrdfLookupTexture(unsigned int size);

template<typename T>
void uploadDataToTexture2D(GLuint texture, GLuint mipMapLevel, GLuint width, GLuint height, GLenum format, GLenum type, const std::vector<T>& buffer)
{
    glTextureSubImage2D(texture, mipMapLevel, 0, 0, width, height, format, type, buffer.data());
}

glm::mat4 getProjectionReversedZInfFar(uint32_t width, uint32_t height, float fovDegrees, float zNear);
glm::mat4 getProjectionReversedZ(uint32_t width, uint32_t height, float fovDegrees, float zNear, float zFar);

std::array<glm::mat4, 6> getCubemapViewMatrices(glm::vec3 cameraPosition);
void updateCameraUBO(UniformBuffer& buffer, glm::mat4 projection, glm::mat4 view, glm::vec3 pos, float nearPlane, float farPlane);

std::string toLowerCase(std::string&& s);

template<typename T>
T uiCeil(T dividend, T divisor)
{
    static_assert(std::is_unsigned<T>());
    return 1U + (dividend - 1U) / divisor;
}

template<typename T, typename U>
T uiCeil(T dividend, U divisor)
{
    static_assert(std::is_unsigned<T>());
    static_assert(std::is_unsigned<U>());
    return 1U + (dividend - 1U) / divisor;
}
}  // namespace spark::utils