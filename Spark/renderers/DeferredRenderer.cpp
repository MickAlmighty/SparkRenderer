#include "DeferredRenderer.hpp"

#include "CommonUtils.h"
#include "ResourceLibrary.h"
#include "Scene.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::renderers
{
DeferredRenderer::DeferredRenderer(unsigned int width, unsigned int height) : Renderer(width, height), gBuffer(width, height)
{
    brdfLookupTexture = utils::createBrdfLookupTexture(1024);

    lightingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("light.glsl");
    createFrameBuffersAndTextures();
}

DeferredRenderer::~DeferredRenderer()
{
    glDeleteTextures(1, &brdfLookupTexture);
    glDeleteTextures(1, &lightingTexture);
    glDeleteFramebuffers(1, &framebuffer);
}

void DeferredRenderer::renderMeshes(const std::shared_ptr<Scene>& scene)
{
    gBuffer.fill(scene->getRenderingQueues(), scene->getCamera()->getUbo());

    GLuint aoTexture{0};
    if(isAmbientOcclusionEnabled)
    {
        aoTexture = ao.process(gBuffer.depthTexture, gBuffer.normalsTexture, scene->getCamera());
    }

    PUSH_DEBUG_GROUP(PBR_LIGHT);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    lightingShader->use();
    lightingShader->bindUniformBuffer("Camera", scene->getCamera()->getUbo());
    lightingShader->bindSSBO("DirLightData", scene->lightManager->getDirLightSSBO());
    lightingShader->bindSSBO("PointLightData", scene->lightManager->getPointLightSSBO());
    lightingShader->bindSSBO("SpotLightData", scene->lightManager->getSpotLightSSBO());
    lightingShader->bindSSBO("LightProbeData", scene->lightManager->getLightProbeSSBO());

    if(const auto cubemap = scene->getSkyboxCubemap().lock(); cubemap)
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
}

void DeferredRenderer::resizeDerived(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
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

GLuint DeferredRenderer::getLightingTexture() const
{
    return lightingTexture;
}
}  // namespace spark::renderers