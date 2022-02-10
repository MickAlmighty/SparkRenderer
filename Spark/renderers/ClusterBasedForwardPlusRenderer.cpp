#include "ClusterBasedForwardPlusRenderer.hpp"

#include "utils/CommonUtils.h"
#include "ICamera.hpp"
#include "Logging.h"
#include "Spark.h"

namespace spark::renderers
{
ClusterBasedForwardPlusRenderer::ClusterBasedForwardPlusRenderer(unsigned int width, unsigned int height,
                                                                 const std::shared_ptr<resources::Shader>& activeClustersDeterminationShader,
                                                                 const std::shared_ptr<resources::Shader>& lightCullingShader)
    : Renderer(width, height), brdfLookupTexture(utils::createBrdfLookupTexture(1024)),
      lightCullingPass(width, height, activeClustersDeterminationShader, lightCullingShader)
{
    depthOnlyShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/depthOnly.glsl");
    depthAndNormalsShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/depthAndNormals.glsl");
    lightingShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/clusterBasedForwardPlusPbrLighting.glsl");
    createFrameBuffersAndTextures();
}

ClusterBasedForwardPlusRenderer::~ClusterBasedForwardPlusRenderer()
{
    glDeleteFramebuffers(1, &lightingFramebuffer);
    glDeleteFramebuffers(1, &depthPrepassFramebuffer);
}

void ClusterBasedForwardPlusRenderer::depthPrepass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    PUSH_DEBUG_GROUP(DEPTH_PREPASS)
    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, depthPrepassFramebuffer);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glClearDepth(0.0);

    std::shared_ptr<resources::Shader> shader{nullptr};
    if(!isAmbientOcclusionEnabled)
    {
        glClear(GL_DEPTH_BUFFER_BIT);
        shader = depthOnlyShader;
    }
    else
    {
        shader = depthAndNormalsShader;
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    }

    shader->use();
    shader->bindUniformBuffer("Camera.camera", camera->getUbo());
    if(const auto it = scene->getRenderingQueues().find(ShaderType::PBR); it != scene->getRenderingQueues().cend())
    {
        for(auto& request : it->second)
        {
            request.mesh->draw(shader, request.model);
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    POP_DEBUG_GROUP()
}

GLuint ClusterBasedForwardPlusRenderer::aoPass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    if(isAmbientOcclusionEnabled)
    {
        return ao.process(depthTexture.get(), normalsTexture.get(), camera);
    }
    return 0;
}

void ClusterBasedForwardPlusRenderer::lightingPass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera,
                                                   const GLuint ssaoTexture)
{
    PUSH_DEBUG_GROUP(PBR_LIGHT)
    glBindFramebuffer(GL_FRAMEBUFFER, lightingFramebuffer);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glDepthFunc(GL_EQUAL);

    if(!scene->getSkyboxCubemap().expired())
    {
        glBindTextureUnit(7, scene->getSkyboxCubemap().lock()->irradianceCubemap.get());
        glBindTextureUnit(8, scene->getSkyboxCubemap().lock()->prefilteredCubemap.get());
    }
    else
    {
        glBindTextureUnit(7, skyboxPlaceholder.get());
        glBindTextureUnit(8, skyboxPlaceholder.get());
    }
    glBindTextureUnit(9, brdfLookupTexture.get());
    glBindTextureUnit(10, ssaoTexture);

    lightingShader->use();
    lightingShader->bindUniformBuffer("Camera.camera", camera->getUbo());
    lightingShader->bindUniformBuffer("AlgorithmData", lightCullingPass.algorithmData);
    lightingShader->bindSSBO("DirLightData", scene->lightManager->getDirLightSSBO());
    lightingShader->bindSSBO("PointLightData", scene->lightManager->getPointLightSSBO());
    lightingShader->bindSSBO("SpotLightData", scene->lightManager->getSpotLightSSBO());
    lightingShader->bindSSBO("LightProbeData", scene->lightManager->getLightProbeSSBO());
    lightingShader->bindSSBO("GlobalPointLightIndices", lightCullingPass.globalPointLightIndices);
    lightingShader->bindSSBO("GlobalSpotLightIndices", lightCullingPass.globalSpotLightIndices);
    lightingShader->bindSSBO("GlobalLightProbeIndices", lightCullingPass.globalLightProbeIndices);
    lightingShader->bindSSBO("PerClusterGlobalLightIndicesBufferMetadata", lightCullingPass.perClusterGlobalLightIndicesBufferMetadata);

    if(const auto it = scene->getRenderingQueues().find(ShaderType::PBR); it != scene->getRenderingQueues().cend())
    {
        for(auto& request : it->second)
        {
            request.mesh->draw(lightingShader, request.model);
        }
    }

    glDepthFunc(GL_GREATER);
    POP_DEBUG_GROUP()
}

void ClusterBasedForwardPlusRenderer::renderMeshes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    if(isProfilingEnabled)
    {
        timer.measure(0, [this, &scene, &camera] { depthPrepass(scene, camera); });
        const GLuint ssaoTexture = aoPass(scene, camera);
        timer.measure(1, [this, &scene, &camera] { lightCullingPass.process(depthTexture.get(), scene, camera); });
        timer.measure(2, [this, &scene, &camera, ssaoTexture] { lightingPass(scene, camera, ssaoTexture); });

        auto m = timer.getMeasurementsInUs();

        SPARK_RENDERER_INFO("{}, {}, {}", m[0], m[1], m[2]);
    }
    else
    {
        depthPrepass(scene, camera);
        const GLuint ssaoTexture = aoPass(scene, camera);
        lightCullingPass.process(depthTexture.get(), scene, camera);
        lightingPass(scene, camera, ssaoTexture);
    }
}

void ClusterBasedForwardPlusRenderer::resizeDerived(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    lightCullingPass.resize(w, h);
    createFrameBuffersAndTextures();
}

GLuint ClusterBasedForwardPlusRenderer::getDepthTexture() const
{
    return depthTexture.get();
}

GLuint ClusterBasedForwardPlusRenderer::getLightingTexture() const
{
    return lightingTexture.get();
}

void ClusterBasedForwardPlusRenderer::createFrameBuffersAndTextures()
{
    lightingTexture = utils::createTexture2D(w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    normalsTexture = utils::createTexture2D(w, h, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    depthTexture = utils::createTexture2D(w, h, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(depthPrepassFramebuffer, {normalsTexture.get()});
    utils::bindDepthTexture(depthPrepassFramebuffer, depthTexture.get());
    utils::recreateFramebuffer(lightingFramebuffer, {lightingTexture.get()});
    utils::bindDepthTexture(lightingFramebuffer, depthTexture.get());
}
}  // namespace spark::renderers