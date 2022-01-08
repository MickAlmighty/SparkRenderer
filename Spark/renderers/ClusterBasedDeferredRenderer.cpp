#include "ClusterBasedDeferredRenderer.hpp"

#include "CommonUtils.h"
#include "ICamera.hpp"
#include "Shader.h"
#include "Spark.h"

namespace spark::renderers
{
ClusterBasedDeferredRenderer::ClusterBasedDeferredRenderer(unsigned int width, unsigned int height)
    : Renderer(width, height), gBuffer(width, height), lightCullingPass(width, height)
{
    brdfLookupTexture = utils::createBrdfLookupTexture(1024);

    lightingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("clusterBasedDeferredPbrLighting.glsl");
    createFrameBuffersAndTextures();
}

ClusterBasedDeferredRenderer::~ClusterBasedDeferredRenderer()
{
    glDeleteTextures(1, &lightingTexture);
}

void ClusterBasedDeferredRenderer::renderMeshes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    gBuffer.fill(scene->getRenderingQueues(), camera->getUbo());

    lightCullingPass.process(gBuffer.depthTexture, scene, camera);

    GLuint ssaoTexture{0};
    if(isAmbientOcclusionEnabled)
        ssaoTexture = ao.process(gBuffer.depthTexture, gBuffer.normalsTexture, camera);

    PUSH_DEBUG_GROUP(TILE_BASED_DEFERRED)
    float clearRgba[] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearTexImage(lightingTexture, 0, GL_RGBA, GL_FLOAT, &clearRgba);

    const auto cubemap = scene->getSkyboxCubemap().lock();

    lightingShader->use();
    lightingShader->setVec2("tileSize", lightCullingPass.pxTileSize);
    lightingShader->bindUniformBuffer("Camera", camera->getUbo());
    lightingShader->bindSSBO("DirLightData", scene->lightManager->getDirLightSSBO());
    lightingShader->bindSSBO("PointLightData", scene->lightManager->getPointLightSSBO());
    lightingShader->bindSSBO("SpotLightData", scene->lightManager->getSpotLightSSBO());
    lightingShader->bindSSBO("LightProbeData", scene->lightManager->getLightProbeSSBO());
    lightingShader->bindSSBO("GlobalPointLightIndices", lightCullingPass.globalPointLightIndices);
    lightingShader->bindSSBO("GlobalSpotLightIndices", lightCullingPass.globalSpotLightIndices);
    lightingShader->bindSSBO("GlobalLightProbeIndices", lightCullingPass.globalLightProbeIndices);
    lightingShader->bindSSBO("PerClusterGlobalLightIndicesBufferMetadata", lightCullingPass.perClusterGlobalLightIndicesBufferMetadata);

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

void ClusterBasedDeferredRenderer::resizeDerived(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    gBuffer.resize(w, h);
    lightCullingPass.resize(w, h);
    createFrameBuffersAndTextures();
}

void ClusterBasedDeferredRenderer::createFrameBuffersAndTextures()
{
    utils::recreateTexture2D(lightingTexture, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
}

GLuint ClusterBasedDeferredRenderer::getDepthTexture() const
{
    return gBuffer.depthTexture;
}

GLuint ClusterBasedDeferredRenderer::getLightingTexture() const
{
    return lightingTexture;
}
}  // namespace spark::renderers
