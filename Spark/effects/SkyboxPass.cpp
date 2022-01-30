#include "SkyboxPass.hpp"

#include "ICamera.hpp"
#include "utils/CommonUtils.h"
#include "PbrCubemapTexture.hpp"
#include "Scene.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
SkyboxPass::SkyboxPass(unsigned int width, unsigned int height) : w(width), h(height)
{
    cubemapShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("cubemap.glsl");
    createFrameBuffersAndTextures();
}

SkyboxPass::~SkyboxPass()
{
    glDeleteFramebuffers(1, &cubemapFramebuffer);
    glDeleteTextures(1, &cubemapTexture);
}

std::optional<GLuint> SkyboxPass::process(GLuint depthTexture, GLuint lightingTexture, const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    const auto cubemapPtr = scene->getSkyboxCubemap().lock();
    if(!cubemapPtr)
        return {};

    PUSH_DEBUG_GROUP(RENDER_CUBEMAP);
    texturePass.process(w, h, lightingTexture, cubemapTexture);
    utils::bindDepthTexture(cubemapFramebuffer, depthTexture);
    renderSkybox(cubemapFramebuffer, w, h, cubemapPtr, camera->getUbo());
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

void SkyboxPass::renderSkybox(GLuint framebuffer, unsigned int fboWidth, unsigned int fboHeight, const std::shared_ptr<PbrCubemapTexture>& cubemapPtr,
                              const UniformBuffer& cameraUbo)
{
    glViewport(0, 0, fboWidth, fboHeight);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_EQUAL);
    cubemapShader->use();
    cubemapShader->bindUniformBuffer("Camera", cameraUbo);

    glBindTextureUnit(0, cubemapPtr->cubemap.get());
    cube.draw();
    glBindTextureUnit(0, 0);
    glDepthFunc(GL_GREATER);
    glDisable(GL_DEPTH_TEST);
}

void SkyboxPass::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createFrameBuffersAndTextures();
}

void SkyboxPass::createFrameBuffersAndTextures()
{
    utils::recreateTexture2D(cubemapTexture, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(cubemapFramebuffer, {cubemapTexture});
}
}  // namespace spark::effects