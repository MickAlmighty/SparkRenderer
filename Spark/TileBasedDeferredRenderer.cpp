#include "TileBasedDeferredRenderer.hpp"

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark
{
TileBasedDeferredRenderer::TileBasedDeferredRenderer(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                                                     const std::shared_ptr<lights::LightManager>& lightManager)
    : Renderer(width, height, cameraUbo), gBuffer(width, height), lightCullingPass(width, height, cameraUbo, lightManager)
{
    brdfLookupTexture = utils::createBrdfLookupTexture(1024);

    tileBasedLightingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("tileBasedLighting.glsl");
    tileBasedLightingShader->bindUniformBuffer("Camera", cameraUbo);
    tileBasedLightingShader->bindSSBO("PointLightIndices", lightCullingPass.pointLightIndices);
    tileBasedLightingShader->bindSSBO("SpotLightIndices", lightCullingPass.spotLightIndices);
    tileBasedLightingShader->bindSSBO("LightProbeIndices", lightCullingPass.lightProbeIndices);
    bindLightBuffers(lightManager);
    createFrameBuffersAndTextures();
}

TileBasedDeferredRenderer::~TileBasedDeferredRenderer()
{
    glDeleteTextures(1, &lightingTexture);
}

GLuint TileBasedDeferredRenderer::process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue,
                                          const std::weak_ptr<PbrCubemapTexture>& pbrCubemap, const UniformBuffer& cameraUbo)
{
    gBuffer.fill(renderQueue, cameraUbo);

    lightCullingPass.process(gBuffer.depthTexture);

    GLuint ssaoTexture{0};
    if(isAmbientOcclusionEnabled)
        ssaoTexture = ao.process(gBuffer.depthTexture, gBuffer.normalsTexture);

    PUSH_DEBUG_GROUP(TILE_BASED_DEFERRED)
    float clearRgba[] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearTexImage(lightingTexture, 0, GL_RGBA, GL_FLOAT, &clearRgba);

    const auto cubemap = pbrCubemap.lock();

    tileBasedLightingShader->use();

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

    tileBasedLightingShader->dispatchCompute(utils::uiCeil(w, 16u), utils::uiCeil(h, 16u), 1);
    glBindTextures(0, 0, nullptr);

    POP_DEBUG_GROUP();
    return lightingTexture;
}

void TileBasedDeferredRenderer::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    gBuffer.resize(w, h);
    ao.resize(w, h);
    lightCullingPass.resize(w, h);
    createFrameBuffersAndTextures();
}

void TileBasedDeferredRenderer::createFrameBuffersAndTextures()
{
    utils::recreateTexture2D(lightingTexture, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
}

void TileBasedDeferredRenderer::bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager)
{
    tileBasedLightingShader->bindSSBO("DirLightData", lightManager->getDirLightSSBO());
    tileBasedLightingShader->bindSSBO("PointLightData", lightManager->getPointLightSSBO());
    tileBasedLightingShader->bindSSBO("SpotLightData", lightManager->getSpotLightSSBO());
    tileBasedLightingShader->bindSSBO("LightProbeData", lightManager->getLightProbeSSBO());
    lightCullingPass.bindLightBuffers(lightManager);
}

GLuint TileBasedDeferredRenderer::getDepthTexture() const
{
    return gBuffer.depthTexture;
}
}  // namespace spark
