#include "CommonUtils.h"

#include "ResourceLibrary.h"
#include "Shader.h"
#include "Spark.h"

namespace spark
{
namespace utils
{
    void createTexture2D(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping,
                       GLenum textureSampling, bool mipMaps, void* data)
    {
        glCreateTextures(GL_TEXTURE_2D, 1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

        if (data == nullptr)
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

    void createCubemap(GLuint& texture, unsigned size, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping,
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

    void bindDepthTexture(GLuint& framebuffer, GLuint depthTexture)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void createFramebuffer(GLuint& framebuffer, const std::vector<GLuint>&& colorTextures, GLuint renderbuffer)
    {
        glCreateFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

        std::vector<GLenum> colorAttachments;
        colorAttachments.reserve(colorTextures.size());
        for(unsigned int i = 0; i < colorTextures.size(); ++i)
        {
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorTextures[i], 0);
            colorAttachments.push_back(GL_COLOR_ATTACHMENT0 + i);
        }
        glDrawBuffers(static_cast<GLsizei>(colorAttachments.size()), colorAttachments.data());

        if(renderbuffer != 0)
        {
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);
        }

        if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        {
            throw std::exception("Framebuffer incomplete!");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    GLuint createBrdfLookupTexture(unsigned size)
    {
        PUSH_DEBUG_GROUP(BRDF_LOOKUP_TABLE);

        GLuint brdfLUTTexture{};
        utils::createTexture2D(brdfLUTTexture, size, size, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

        const auto brdfComputeShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("brdfCompute.glsl");
        brdfComputeShader->use();
        glBindImageTexture(0, brdfLUTTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16F);
        brdfComputeShader->dispatchCompute(size / 32, size / 32, 1);

        POP_DEBUG_GROUP();

        return brdfLUTTexture;
    }
}  // namespace utils
}  // namespace spark
