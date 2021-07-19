#include "CommonUtils.h"

#include <numeric>

#include <glm/gtc/matrix_transform.hpp>

#include "Logging.h"
#include "ResourceLibrary.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::utils
{
void createTexture2D(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping,
                     GLenum textureSampling, bool mipMaps, void* data)
{
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    if(data == nullptr)
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, pixelFormat, nullptr);
    else
        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, pixelFormat, data);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, textureSampling);

    if(mipMaps)
    {
        glGenerateMipmap(GL_TEXTURE_2D);
        if(textureSampling == GL_LINEAR)
        {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        }
        else if(textureSampling == GL_NEAREST)
        {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        }
    }
    else
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, textureSampling);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, textureWrapping);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, textureWrapping);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void recreateTexture2D(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping,
                       GLenum textureSampling, bool mipMaps, void* data)
{
    if(texture != 0)
        glDeleteTextures(1, &texture);

    createTexture2D(texture, width, height, internalFormat, format, pixelFormat, textureWrapping, textureSampling, mipMaps, data);
}

void createCubemap(GLuint& texture, unsigned int size, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping,
                   GLenum textureSampling, bool mipMaps)
{
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
    for(unsigned int i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, internalFormat, size, size, 0, format, pixelFormat, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, textureWrapping);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, textureWrapping);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, textureWrapping);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, textureSampling);
    if(mipMaps)
    {
        if(textureSampling == GL_LINEAR)
        {
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        }
        else if(textureSampling == GL_NEAREST)
        {
            glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        }
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, textureSampling);
    }
}

void recreateCubemap(GLuint& texture, unsigned int size, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping,
                     GLenum textureSampling, bool mipMaps)
{
    if(texture != 0)
        glDeleteTextures(1, &texture);

    createCubemap(texture, size, internalFormat, format, pixelFormat, textureWrapping, textureSampling, mipMaps);
}

void bindTexture2D(GLuint framebuffer, GLuint colorTexture, GLuint renderTargetIds, GLuint mipmapLevel)
{
    glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0 + renderTargetIds, colorTexture, mipmapLevel);
    glNamedFramebufferDrawBuffer(framebuffer, GL_COLOR_ATTACHMENT0 + renderTargetIds);
}

void bindTextures2D(GLuint framebuffer, const std::vector<GLuint>& colorTextures, const std::vector<GLuint>& renderTargetIds)
{
    if(colorTextures.empty())
    {
        return;
    }

    if(colorTextures.size() != renderTargetIds.size())
    {
        std::vector<GLenum> attachments(colorTextures.size());
        std::iota(attachments.begin(), attachments.end(), GL_COLOR_ATTACHMENT0);

        for(unsigned int i = 0; i < colorTextures.size(); ++i)
        {
            glNamedFramebufferTexture(framebuffer, attachments[i], colorTextures[i], 0);
        }

        glNamedFramebufferDrawBuffers(framebuffer, static_cast<GLsizei>(attachments.size()), attachments.data());
    }
    else
    {
        std::vector<GLenum> attachments(colorTextures.size());
        for(unsigned int i = 0; i < colorTextures.size(); ++i)
        {
            attachments[i] = GL_COLOR_ATTACHMENT0 + renderTargetIds[i];
            glNamedFramebufferTexture(framebuffer, attachments[i], colorTextures[i], 0);
        }
        glNamedFramebufferDrawBuffers(framebuffer, static_cast<GLsizei>(attachments.size()), attachments.data());
    }
}

void bindTextures2D(GLuint framebuffer, const std::vector<GLuint>&& colorTextures, const std::vector<GLuint>&& renderTargetIds)
{
    bindTextures2D(framebuffer, colorTextures, renderTargetIds);
}

void bindDepthTexture(GLuint& framebuffer, GLuint depthTexture)
{
    glNamedFramebufferTexture(framebuffer, GL_DEPTH_ATTACHMENT, depthTexture, 0);
}

void createFramebuffer(GLuint& framebuffer, std::vector<GLuint>&& colorTextures, GLuint renderbuffer)
{
    glCreateFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    if(!colorTextures.empty())
    {
        bindTextures2D(framebuffer, colorTextures);
    }

    if(renderbuffer != 0)
    {
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);
    }

    if(!colorTextures.empty() || renderbuffer != 0)
    {
        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            throw std::runtime_error("Framebuffer incomplete!");
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void recreateFramebuffer(GLuint& framebuffer, std::vector<GLuint>&& colorTextures, GLuint renderbuffer)
{
    if(framebuffer != 0)
        glDeleteFramebuffers(1, &framebuffer);

    createFramebuffer(framebuffer, std::move(colorTextures), renderbuffer);
}

GLuint createBrdfLookupTexture(unsigned int size)
{
    PUSH_DEBUG_GROUP(BRDF_LOOKUP_TABLE);

    GLuint brdfLUTTexture{};
    utils::createTexture2D(brdfLUTTexture, size, size, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    const auto brdfComputeShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("brdfCompute.glsl");
    brdfComputeShader->use();
    glBindImageTexture(0, brdfLUTTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);
    brdfComputeShader->dispatchCompute(size / 32, size / 32, 1);

    POP_DEBUG_GROUP();

    return brdfLUTTexture;
}

glm::mat4 getProjectionReversedZInfFar(uint32_t width, uint32_t height, float fovDegrees, float zNear)
{
    const float aspectWbyH = static_cast<float>(width) / (1.0f * height);
    const float f = 1.0f / glm::tan(glm::radians(fovDegrees) / 2.0f);

    const glm::vec4 column0{f / aspectWbyH, 0.0f, 0.0f, 0.0f};
    const glm::vec4 column1{0.0f, f, 0.0f, 0.0f};
    const glm::vec4 column2{0.0f, 0.0f, 0.0f, -1.0f};
    const glm::vec4 column3{0.0f, 0.0f, zNear, 0.0f};

    return glm::mat4(column0, column1, column2, column3);
}

glm::mat4 getProjectionReversedZ(uint32_t width, uint32_t height, float fovDegrees, float zNear, float zFar)
{
    const float rad = glm::radians(fovDegrees);
    const float h = glm::cos(0.5f * rad) / glm::sin(0.5f * rad);
    const float w = h * static_cast<float>(height) / (1.0f * width);

    const glm::vec4 column0{w, 0.0f, 0.0f, 0.0f};
    const glm::vec4 column1{0.0f, h, 0.0f, 0.0f};
    const glm::vec4 column2{0.0f, 0.0f, zNear / (zFar - zNear), -1.0f};
    const glm::vec4 column3{0.0f, 0.0f, -(zFar * zNear) / (zNear - zFar), 0.0f};

    return glm::mat4(column0, column1, column2, column3);
}

std::array<glm::mat4, 6> getCubemapViewMatrices(glm::vec3 cameraPosition)
{
    return {glm::lookAt(cameraPosition, cameraPosition + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
            glm::lookAt(cameraPosition, cameraPosition + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
            glm::lookAt(cameraPosition, cameraPosition + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
            glm::lookAt(cameraPosition, cameraPosition + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
            glm::lookAt(cameraPosition, cameraPosition + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
            glm::lookAt(cameraPosition, cameraPosition + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))};
}
}  // namespace spark::utils
