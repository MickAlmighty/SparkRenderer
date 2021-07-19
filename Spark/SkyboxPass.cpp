#include "SkyboxPass.hpp"

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark
{
SkyboxPass::~SkyboxPass()
{
    cleanup();
}

void SkyboxPass::setup(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo)
{
    w = width;
    h = height;
    cubemapShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("cubemap.glsl");
    cubemapShader->bindUniformBuffer("Camera", cameraUbo);
}

std::optional<GLuint> SkyboxPass::process(const std::weak_ptr<PbrCubemapTexture>& cubemap, GLuint depthTexture, GLuint lightingTexture)
{
    const auto cubemapPtr = cubemap.lock();
    if(!cubemapPtr)
        return {};

    PUSH_DEBUG_GROUP(RENDER_CUBEMAP);
    utils::bindTexture2D(cubemapFramebuffer, lightingTexture);
    utils::bindDepthTexture(cubemapFramebuffer, depthTexture);
    renderSkybox(cubemapFramebuffer, w, h, cubemapPtr);
    POP_DEBUG_GROUP()
    return lightingTexture;
}

void SkyboxPass::processFramebuffer(const std::weak_ptr<PbrCubemapTexture>& cubemap, GLuint framebuffer, unsigned int fboWidth,
                                    unsigned int fboHeight)
{
    const auto cubemapPtr = cubemap.lock();
    if(!cubemapPtr)
        return;

    PUSH_DEBUG_GROUP(RENDER_CUBEMAP);
    renderSkybox(framebuffer, fboWidth, fboHeight, cubemapPtr);
    POP_DEBUG_GROUP()
}

void SkyboxPass::renderSkybox(GLuint framebuffer, unsigned fboWidth, unsigned fboHeight, const std::shared_ptr<PbrCubemapTexture>& cubemapPtr)
{
    glViewport(0, 0, fboWidth, fboHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GEQUAL);
    cubemapShader->use();

    glBindTextureUnit(0, cubemapPtr->cubemap);
    cube.draw();
    glBindTextureUnit(0, 0);
    glDepthFunc(GL_GREATER);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, w, h);
}

void SkyboxPass::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    utils::recreateFramebuffer(cubemapFramebuffer, {});
}

void SkyboxPass::cleanup()
{
    glDeleteFramebuffers(1, &cubemapFramebuffer);
}
}  // namespace spark