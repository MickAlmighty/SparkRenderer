#include "CommonUtils.h"


namespace spark
{
namespace utils
{
    void createTexture(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat, GLenum textureWrapping,
                       GLenum textureSampling, bool mipMaps)
    {
        glCreateTextures(GL_TEXTURE_2D, 1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);

        glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, pixelFormat, 0);
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
}
}