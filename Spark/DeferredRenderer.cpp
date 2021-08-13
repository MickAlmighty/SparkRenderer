#include "DeferredRenderer.hpp"

#include "CommonUtils.h"
#include "ResourceLibrary.h"
#include "Shader.h"
#include "Spark.h"

namespace spark
{
DeferredRenderer::DeferredRenderer(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                                   const std::shared_ptr<lights::LightManager>& lightManager)
    : Renderer(width, height, cameraUbo), gBuffer(width, height)
{
    brdfLookupTexture = utils::createBrdfLookupTexture(1024);

    lightingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("light.glsl");
    lightingShader->bindUniformBuffer("Camera", cameraUbo);
    bindLightBuffers(lightManager);
    createFrameBuffersAndTextures();
}

DeferredRenderer::~DeferredRenderer()
{
    glDeleteTextures(1, &brdfLookupTexture);
    glDeleteTextures(1, &lightingTexture);
    glDeleteFramebuffers(1, &framebuffer);
}

GLuint DeferredRenderer::process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::weak_ptr<PbrCubemapTexture>& pbrCubemap,
                                 const UniformBuffer& cameraUbo)
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
    lightingShader->bindSSBO("LightProbeData", lightManager->getLightProbeSSBO());
}

void DeferredRenderer::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    ao.resize(w, h);
    gBuffer.resize(w, h);
    createFrameBuffersAndTextures();
}

void DeferredRenderer::createFrameBuffersAndTextures()
{
    utils::recreateTexture2D(lightingTexture, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(framebuffer, {lightingTexture});
}

GLuint DeferredRenderer::getDepthTexture() const
{
    return gBuffer.depthTexture;
}
}  // namespace spark