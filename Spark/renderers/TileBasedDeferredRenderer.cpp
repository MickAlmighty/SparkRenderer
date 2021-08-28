#include "TileBasedDeferredRenderer.hpp"

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::renderers
{
TileBasedDeferredRenderer::TileBasedDeferredRenderer(unsigned int width, unsigned int height)
    : Renderer(width, height), gBuffer(width, height), lightCullingPass(width, height)
{
    brdfLookupTexture = utils::createBrdfLookupTexture(1024);

    lightingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("tileBasedLighting.glsl");
    lightingShader->bindSSBO("PointLightIndices", lightCullingPass.pointLightIndices);
    lightingShader->bindSSBO("SpotLightIndices", lightCullingPass.spotLightIndices);
    lightingShader->bindSSBO("LightProbeIndices", lightCullingPass.lightProbeIndices);
    createFrameBuffersAndTextures();
}

TileBasedDeferredRenderer::~TileBasedDeferredRenderer()
{
    glDeleteTextures(1, &lightingTexture);
}

void TileBasedDeferredRenderer::renderMeshes(const std::shared_ptr<Scene>& scene)
{
    gBuffer.fill(scene->getRenderingQueues(), scene->getCamera()->getUbo());

    lightCullingPass.process(gBuffer.depthTexture, scene);

    GLuint ssaoTexture{0};
    if(isAmbientOcclusionEnabled)
        ssaoTexture = ao.process(gBuffer.depthTexture, gBuffer.normalsTexture, scene->getCamera());

    PUSH_DEBUG_GROUP(TILE_BASED_DEFERRED)
    float clearRgba[] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearTexImage(lightingTexture, 0, GL_RGBA, GL_FLOAT, &clearRgba);

    const auto cubemap = scene->getSkyboxCubemap().lock();

    lightingShader->use();
    lightingShader->bindUniformBuffer("Camera", scene->getCamera()->getUbo());
    lightingShader->bindSSBO("DirLightData", scene->lightManager->getDirLightSSBO());
    lightingShader->bindSSBO("PointLightData", scene->lightManager->getPointLightSSBO());
    lightingShader->bindSSBO("SpotLightData", scene->lightManager->getSpotLightSSBO());
    lightingShader->bindSSBO("LightProbeData", scene->lightManager->getLightProbeSSBO());

    // depth texture as sampler2D
    glBindTextureUnit(0, gBuffer.depthTexture);
    if(cubemap)
    {
        glBindTextureUnit(1, cubemap->irradianceCubemap);
        glBindTextureUnit(2, cubemap->prefilteredCubemap);
    }
    glBindTextureUnit(3, brdfLookupTexture);
    glBindTextureUnit(4, ssaoTexture);

    // textures as images
    glBindImageTexture(0, gBuffer.colorTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA8);
    glBindImageTexture(1, gBuffer.normalsTexture, 0, false, 0, GL_READ_ONLY, GL_RG16F);
    glBindImageTexture(2, gBuffer.roughnessMetalnessTexture, 0, false, 0, GL_READ_ONLY, GL_RG8);

    // output image
    glBindImageTexture(3, lightingTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

    lightingShader->dispatchCompute(utils::uiCeil(w, 16u), utils::uiCeil(h, 16u), 1);
    glBindTextures(0, 0, nullptr);

    POP_DEBUG_GROUP();
}

void TileBasedDeferredRenderer::resizeDerived(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    gBuffer.resize(w, h);
    lightCullingPass.resize(w, h);
    createFrameBuffersAndTextures();
}

void TileBasedDeferredRenderer::createFrameBuffersAndTextures()
{
    utils::recreateTexture2D(lightingTexture, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
}

GLuint TileBasedDeferredRenderer::getDepthTexture() const
{
    return gBuffer.depthTexture;
}

GLuint TileBasedDeferredRenderer::getLightingTexture() const
{
    return lightingTexture;
}
}  // namespace spark::renderers
