#include "TileBasedDeferredRenderer.hpp"

#include "utils/CommonUtils.h"
#include "ICamera.hpp"
#include "Logging.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::renderers
{
TileBasedDeferredRenderer::TileBasedDeferredRenderer(unsigned int width, unsigned int height)
    : Renderer(width, height), brdfLookupTexture(utils::createBrdfLookupTexture(1024)), gBuffer(width, height), lightCullingPass(width, height)
{
    lightingShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/tileBasedLighting.glsl");
    lightingShader->bindSSBO("PointLightIndices", lightCullingPass.pointLightIndices);
    lightingShader->bindSSBO("SpotLightIndices", lightCullingPass.spotLightIndices);
    lightingShader->bindSSBO("LightProbeIndices", lightCullingPass.lightProbeIndices);
    createFrameBuffersAndTextures();
}

void TileBasedDeferredRenderer::processLighting(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera, GLuint ssaoTexture)
{
    PUSH_DEBUG_GROUP(TILE_BASED_DEFERRED)
    float clearRgba[] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearTexImage(lightingTexture.get(), 0, GL_RGBA, GL_FLOAT, &clearRgba);

    lightingShader->use();
    lightingShader->bindUniformBuffer("Camera", camera->getUbo());
    lightingShader->bindSSBO("DirLightData", scene->lightManager->getDirLightSSBO());
    lightingShader->bindSSBO("PointLightData", scene->lightManager->getPointLightSSBO());
    lightingShader->bindSSBO("SpotLightData", scene->lightManager->getSpotLightSSBO());
    lightingShader->bindSSBO("LightProbeData", scene->lightManager->getLightProbeSSBO());

    // depth texture as sampler2D
    glBindTextureUnit(0, gBuffer.depthTexture.get());
    if(const auto cubemap = scene->getSkyboxCubemap().lock(); cubemap)
    {
        glBindTextureUnit(1, cubemap->irradianceCubemap.get());
        glBindTextureUnit(2, cubemap->prefilteredCubemap.get());
    }
    glBindTextureUnit(3, brdfLookupTexture.get());
    glBindTextureUnit(4, ssaoTexture);

    // textures as images
    glBindImageTexture(0, gBuffer.colorTexture.get(), 0, false, 0, GL_READ_ONLY, GL_RGBA8);
    glBindImageTexture(1, gBuffer.normalsTexture.get(), 0, false, 0, GL_READ_ONLY, GL_RG16F);
    glBindImageTexture(2, gBuffer.roughnessMetalnessTexture.get(), 0, false, 0, GL_READ_ONLY, GL_RG8);

    // output image
    glBindImageTexture(3, lightingTexture.get(), 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

    lightingShader->dispatchCompute(utils::uiCeil(w, 16u), utils::uiCeil(h, 16u), 1);
    glBindTextures(0, 0, nullptr);

    POP_DEBUG_GROUP();
}

GLuint TileBasedDeferredRenderer::aoPass(const std::shared_ptr<ICamera>& camera)
{
    if(isAmbientOcclusionEnabled)
        return ao.process(gBuffer.depthTexture.get(), gBuffer.normalsTexture.get(), camera);

    return 0;
}

void TileBasedDeferredRenderer::renderMeshes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    if(isProfilingEnabled)
    {
        timer.measure(0, [this, &scene, &camera] { gBuffer.fill(scene->getRenderingQueues(), camera->getUbo()); });
        timer.measure(1, [this, &scene, &camera] { lightCullingPass.process(gBuffer.depthTexture.get(), scene, camera); });
        const GLuint ssaoTexture = aoPass(camera);
        timer.measure(2, [this, &scene, &camera, ssaoTexture] { processLighting(scene, camera, ssaoTexture); });

        auto m = timer.getMeasurementsInUs();

        SPARK_RENDERER_INFO("{}, {}, {}", m[0], m[1], m[2]);
    }
    else
    {
        gBuffer.fill(scene->getRenderingQueues(), camera->getUbo());
        lightCullingPass.process(gBuffer.depthTexture.get(), scene, camera);
        const GLuint ssaoTexture = aoPass(camera);
        processLighting(scene, camera, ssaoTexture);
    }
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
    lightingTexture = utils::createTexture2D(w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
}

GLuint TileBasedDeferredRenderer::getDepthTexture() const
{
    return gBuffer.depthTexture.get();
}

GLuint TileBasedDeferredRenderer::getLightingTexture() const
{
    return lightingTexture.get();
}
}  // namespace spark::renderers
