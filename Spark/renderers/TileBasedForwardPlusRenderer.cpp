#include "TileBasedForwardPlusRenderer.hpp"

#include "CommonUtils.h"
#include "ICamera.hpp"
#include "Logging.h"
#include "Spark.h"

namespace spark::renderers
{
TileBasedForwardPlusRenderer::TileBasedForwardPlusRenderer(unsigned int width, unsigned int height)
    : Renderer(width, height), lightCullingPass(width, height)
{
    brdfLookupTexture = utils::createBrdfLookupTexture(1024);

    depthOnlyShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("depthOnly.glsl");
    depthAndNormalsShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("depthAndNormals.glsl");
    lightingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("tileBasedForwardPlusPbrLighting.glsl");

    createFrameBuffersAndTextures();
}

TileBasedForwardPlusRenderer::~TileBasedForwardPlusRenderer()
{
    glDeleteTextures(1, &lightingTexture);
    glDeleteTextures(1, &normalsTexture);
    glDeleteTextures(1, &depthTexture);
    glDeleteTextures(1, &brdfLookupTexture);
    glDeleteFramebuffers(1, &lightingFramebuffer);
    glDeleteFramebuffers(1, &depthPrepassFramebuffer);
}

void TileBasedForwardPlusRenderer::depthPrepass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
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
    shader->bindUniformBuffer("Camera", camera->getUbo());
    if(const auto it = scene->getRenderingQueues().find(ShaderType::PBR); it != scene->getRenderingQueues().cend())
    {
        for(auto& request : it->second)
        {
            request.mesh->draw(shader, request.model);
        }
    }

    POP_DEBUG_GROUP()
}

GLuint TileBasedForwardPlusRenderer::aoPass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    if(isAmbientOcclusionEnabled)
    {
        return ao.process(depthTexture, normalsTexture, camera);
    }
    return 0;
}

void TileBasedForwardPlusRenderer::lightingPass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera, const GLuint ssaoTexture)
{
    PUSH_DEBUG_GROUP(PBR_LIGHT)
    glBindFramebuffer(GL_FRAMEBUFFER, lightingFramebuffer);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glDepthFunc(GL_EQUAL);

    if(!scene->getSkyboxCubemap().expired())
    {
        glBindTextureUnit(7, scene->getSkyboxCubemap().lock()->irradianceCubemap);
        glBindTextureUnit(8, scene->getSkyboxCubemap().lock()->prefilteredCubemap);
    }
    else
    {
        glBindTextures(7, 2, nullptr);
    }
    glBindTextureUnit(9, brdfLookupTexture);
    glBindTextureUnit(10, ssaoTexture);

    lightingShader->use();
    lightingShader->bindUniformBuffer("Camera", camera->getUbo());
    lightingShader->bindSSBO("DirLightData", scene->lightManager->getDirLightSSBO());
    lightingShader->bindSSBO("PointLightData", scene->lightManager->getPointLightSSBO());
    lightingShader->bindSSBO("SpotLightData", scene->lightManager->getSpotLightSSBO());
    lightingShader->bindSSBO("LightProbeData", scene->lightManager->getLightProbeSSBO());
    lightingShader->bindSSBO("PointLightIndices", lightCullingPass.pointLightIndices);
    lightingShader->bindSSBO("SpotLightIndices", lightCullingPass.spotLightIndices);
    lightingShader->bindSSBO("LightProbeIndices", lightCullingPass.lightProbeIndices);
    lightingShader->setUVec2("viewportSize", {w, h});

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

void TileBasedForwardPlusRenderer::renderMeshes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    if(isProfilingEnabled)
    {
        timer.measure(0, [this, &scene, &camera] { depthPrepass(scene, camera); });
        const GLuint ssaoTexture = aoPass(scene, camera);
        timer.measure(1, [this, &scene, &camera] { lightCullingPass.process(depthTexture, scene, camera); });
        timer.measure(2, [this, &scene, &camera, ssaoTexture] { lightingPass(scene, camera, ssaoTexture); });

        auto m = timer.getMeasurementsInUs();

        SPARK_RENDERER_INFO("{}, {}, {}", m[0], m[1], m[2]);
    }
    else
    {
        depthPrepass(scene, camera);
        const GLuint ssaoTexture = aoPass(scene, camera);
        lightCullingPass.process(depthTexture, scene, camera);
        lightingPass(scene, camera, ssaoTexture);
    }
}

void TileBasedForwardPlusRenderer::resizeDerived(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    lightCullingPass.resize(w, h);
    createFrameBuffersAndTextures();
}

GLuint TileBasedForwardPlusRenderer::getDepthTexture() const
{
    return depthTexture;
}

GLuint TileBasedForwardPlusRenderer::getLightingTexture() const
{
    return lightingTexture;
}

void TileBasedForwardPlusRenderer::createFrameBuffersAndTextures()
{
    utils::recreateTexture2D(lightingTexture, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(normalsTexture, w, h, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(depthTexture, w, h, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(depthPrepassFramebuffer, {normalsTexture});
    utils::bindDepthTexture(depthPrepassFramebuffer, depthTexture);
    utils::recreateFramebuffer(lightingFramebuffer, {lightingTexture});
    utils::bindDepthTexture(lightingFramebuffer, depthTexture);
}
}  // namespace spark::renderers