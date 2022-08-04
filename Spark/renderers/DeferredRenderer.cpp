#include "DeferredRenderer.hpp"

#include "utils/CommonUtils.h"
#include "ICamera.hpp"
#include "Logging.h"
#include "ResourceLibrary.h"
#include "Scene.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::renderers
{
DeferredRenderer::DeferredRenderer(unsigned int width, unsigned int height)
    : Renderer(width, height), gBuffer(width, height), brdfLookupTexture(utils::createBrdfLookupTexture(1024))
{
    lightingShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/light.glsl");
    createFrameBuffersAndTextures();
}

DeferredRenderer::~DeferredRenderer()
{
    glDeleteFramebuffers(1, &framebuffer);
}

void DeferredRenderer::processLighting(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera, GLuint aoTexture)
{
    PUSH_DEBUG_GROUP(PBR_LIGHT);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    lightingShader->use();
    lightingShader->bindUniformBuffer("Camera", camera->getUbo());
    lightingShader->bindSSBO("DirLightData", scene->lightManager->getDirLightSSBO());
    lightingShader->bindSSBO("PointLightData", scene->lightManager->getPointLightSSBO());
    lightingShader->bindSSBO("SpotLightData", scene->lightManager->getSpotLightSSBO());
    lightingShader->bindSSBO("LightProbeData", scene->lightManager->getLightProbeSSBO());

    std::array<GLuint, 10> textures{gBuffer.depthTexture.get(),
                                    gBuffer.colorTexture.get(),
                                    gBuffer.normalsTexture.get(),
                                    gBuffer.roughnessMetalnessTexture.get(),
                                    skyboxPlaceholder.get(),
                                    skyboxPlaceholder.get(),
                                    brdfLookupTexture.get(),
                                    aoTexture,
                                    scene->lightManager->getLightProbeManager().getIrradianceCubemapArray(),
                                    scene->lightManager->getLightProbeManager().getPrefilterCubemapArray()};

    if(const auto cubemap = scene->getSkyboxCubemap().lock(); cubemap)
    {
        textures[4] = cubemap->irradianceCubemap.get();
        textures[5] = cubemap->prefilteredCubemap.get();
    }

    glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
    screenQuad.draw();
    glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);

    POP_DEBUG_GROUP();
}

void DeferredRenderer::renderMeshes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    if(isProfilingEnabled)
    {
        timer.measure(0, [this, scene, camera] { gBuffer.fill(scene->getRenderingQueues(), camera->getUbo()); });
        const GLuint aoTexture = processAo(camera);
        timer.measure(1, [this, scene, camera, &aoTexture] { processLighting(scene, camera, aoTexture); });

        auto m = timer.getMeasurementsInUs();
        SPARK_RENDERER_INFO("{}, {}", m[0], m[1]);
    }
    else
    {
        gBuffer.fill(scene->getRenderingQueues(), camera->getUbo());
        const GLuint aoTexture = processAo(camera);
        processLighting(scene, camera, aoTexture);
    }
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
    lightingTexture = utils::createTexture2D(w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(framebuffer, {lightingTexture.get()});
}

GLuint DeferredRenderer::processAo(const std::shared_ptr<ICamera>& camera)
{
    if(isAmbientOcclusionEnabled)
    {
        return ao.process(gBuffer.depthTexture.get(), gBuffer.normalsTexture.get(), camera);
    }
    return 0;
}

GLuint DeferredRenderer::getDepthTexture() const
{
    return gBuffer.depthTexture.get();
}

GLuint DeferredRenderer::getLightingTexture() const
{
    return lightingTexture.get();
}
}  // namespace spark::renderers