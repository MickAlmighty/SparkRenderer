#include "DeferredRenderer.hpp"

#include "CommonUtils.h"
#include "ResourceLibrary.h"
#include "Shader.h"
#include "Spark.h"

namespace spark
{
DeferredRenderer::~DeferredRenderer()
{
    cleanup();
}

void DeferredRenderer::setup(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                             const std::shared_ptr<lights::LightManager>& lightManager)
{
    ao.setup(width, height, cameraUbo);
    brdfLookupTexture = utils::createBrdfLookupTexture(1024);

    lightingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("light.glsl");
    lightingShader->bindUniformBuffer("Camera", cameraUbo);
    bindLightBuffers(lightManager);
}

GLuint DeferredRenderer::process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap, const UniformBuffer& cameraUbo)
{
    gBuffer.fill(renderQueue, cameraUbo);

    GLuint aoTexture{0};
    if(isAmbientOcclusionEnabled)
    {
        aoTexture = ao.process(gBuffer.depthTexture, gBuffer.normalsTexture);
    }

    PUSH_DEBUG_GROUP(PBR_LIGHT);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    lightingShader->use();
    if(const auto cubemap = pbrCubemap.lock(); cubemap)
    {
        std::array<GLuint, 8> textures{
            gBuffer.depthTexture,       gBuffer.colorTexture,        gBuffer.normalsTexture, gBuffer.roughnessMetalnessTexture,
            cubemap->irradianceCubemap, cubemap->prefilteredCubemap, brdfLookupTexture,      aoTexture};
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
    }
    else
    {
        std::array<GLuint, 8> textures{
            gBuffer.depthTexture, gBuffer.colorTexture, gBuffer.normalsTexture, gBuffer.roughnessMetalnessTexture, 0, 0, 0, aoTexture};
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
    }

    POP_DEBUG_GROUP();

    return lightingTexture;
}

void DeferredRenderer::bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager)
{
    lightingShader->bindSSBO("DirLightData", lightManager->getDirLightSSBO());
    lightingShader->bindSSBO("PointLightData", lightManager->getPointLightSSBO());
    lightingShader->bindSSBO("SpotLightData", lightManager->getSpotLightSSBO());
}

void DeferredRenderer::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    gBuffer.createFrameBuffersAndTextures(width, height);
    ao.createFrameBuffersAndTextures(width, height);

    utils::recreateTexture2D(lightingTexture, width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(framebuffer, {lightingTexture});
}

void DeferredRenderer::cleanup()
{
    glDeleteTextures(1, &brdfLookupTexture);
    glDeleteTextures(1, &lightingTexture);
    glDeleteFramebuffers(1, &framebuffer);
    ao.cleanup();
}

GLuint DeferredRenderer::getDepthTexture() const
{
    return gBuffer.depthTexture;
}
}  // namespace spark