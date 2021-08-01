#include "SkyboxPass.hpp"

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
SkyboxPass::~SkyboxPass()
{
    glDeleteFramebuffers(1, &cubemapFramebuffer);
    glDeleteTextures(1, &cubemapTexture);
}

void SkyboxPass::setup(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    cubemapShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("cubemap.glsl");
}

std::optional<GLuint> SkyboxPass::process(const std::weak_ptr<PbrCubemapTexture>& cubemap, GLuint depthTexture, GLuint lightingTexture, const UniformBuffer& cameraUbo)
{
    const auto cubemapPtr = cubemap.lock();
    if(!cubemapPtr)
        return {};

    PUSH_DEBUG_GROUP(RENDER_CUBEMAP);
    texturePass.process(w, h, lightingTexture, cubemapTexture);
    utils::bindDepthTexture(cubemapFramebuffer, depthTexture);
    renderSkybox(cubemapFramebuffer, w, h, cubemapPtr, cameraUbo);
    POP_DEBUG_GROUP()
    return cubemapTexture;
}

void SkyboxPass::processFramebuffer(const std::weak_ptr<PbrCubemapTexture>& cubemap, GLuint framebuffer, unsigned int fboWidth,
                                    unsigned int fboHeight, const UniformBuffer& cameraUbo)
{
    const auto cubemapPtr = cubemap.lock();
    if(!cubemapPtr)
        return;

    PUSH_DEBUG_GROUP(RENDER_CUBEMAP);
    renderSkybox(framebuffer, fboWidth, fboHeight, cubemapPtr, cameraUbo);
    POP_DEBUG_GROUP()
}

void SkyboxPass::renderSkybox(GLuint framebuffer, unsigned int fboWidth, unsigned int fboHeight, const std::shared_ptr<PbrCubemapTexture>& cubemapPtr, const UniformBuffer& cameraUbo)
{
    glViewport(0, 0, fboWidth, fboHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GEQUAL);
    cubemapShader->use();
    cubemapShader->bindUniformBuffer("Camera", cameraUbo);

    glBindTextureUnit(0, cubemapPtr->cubemap);
    cube.draw();
    glBindTextureUnit(0, 0);
    glDepthFunc(GL_GREATER);
    glDisable(GL_DEPTH_TEST);
}

void SkyboxPass::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    utils::recreateTexture2D(cubemapTexture, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(cubemapFramebuffer, {cubemapTexture});
}
}  // namespace spark::effects