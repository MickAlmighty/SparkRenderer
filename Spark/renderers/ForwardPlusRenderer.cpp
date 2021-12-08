#include "ForwardPlusRenderer.hpp"

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::renderers
{
ForwardPlusRenderer::ForwardPlusRenderer(unsigned int width, unsigned int height) : Renderer(width, height)
{
    brdfLookupTexture = utils::createBrdfLookupTexture(1024);

    depthOnlyShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("depthOnly.glsl");
    depthAndNormalsShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("depthAndNormals.glsl");
    lightingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("forwardPlusPbrLighting.glsl");

    createFrameBuffersAndTextures();
}

ForwardPlusRenderer::~ForwardPlusRenderer()
{
    glDeleteTextures(1, &lightingTexture);
    glDeleteTextures(1, &normalsTexture);
    glDeleteTextures(1, &depthTexture);
    glDeleteTextures(1, &brdfLookupTexture);
    glDeleteFramebuffers(1, &lightingFramebuffer);
    glDeleteFramebuffers(1, &depthPrepassFramebuffer);
}

void ForwardPlusRenderer::depthPrepass(const std::shared_ptr<Scene>& scene)
{
    PUSH_DEBUG_GROUP(DEPTH_PREPASS)
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
    shader->bindUniformBuffer("Camera", scene->getCamera()->getUbo());
    if (const auto it = scene->getRenderingQueues().find(ShaderType::PBR); it != scene->getRenderingQueues().cend())
    {
        for (auto& request : it->second)
        {
            request.mesh->draw(shader, request.model);
        }
    }

    POP_DEBUG_GROUP()
}

GLuint ForwardPlusRenderer::aoPass(const std::shared_ptr<Scene>& scene)
{
    if(isAmbientOcclusionEnabled)
    {
        return ao.process(depthTexture, normalsTexture, scene->getCamera());
    }
    return 0;
}

void ForwardPlusRenderer::lightingPass(const std::shared_ptr<Scene>& scene, const GLuint ssaoTexture)
{
    PUSH_DEBUG_GROUP(PBR_LIGHT);
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
    lightingShader->bindUniformBuffer("Camera", scene->getCamera()->getUbo());
    lightingShader->bindSSBO("DirLightData", scene->lightManager->getDirLightSSBO());
    lightingShader->bindSSBO("PointLightData", scene->lightManager->getPointLightSSBO());
    lightingShader->bindSSBO("SpotLightData", scene->lightManager->getSpotLightSSBO());
    lightingShader->bindSSBO("LightProbeData", scene->lightManager->getLightProbeSSBO());

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

void ForwardPlusRenderer::renderMeshes(const std::shared_ptr<Scene>& scene)
{
    depthPrepass(scene);
    const GLuint ssaoTexture = aoPass(scene);
    lightingPass(scene, ssaoTexture);
}

void ForwardPlusRenderer::resizeDerived(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createFrameBuffersAndTextures();
}

GLuint ForwardPlusRenderer::getDepthTexture() const
{
    return depthTexture;
}

GLuint ForwardPlusRenderer::getLightingTexture() const
{
    return lightingTexture;
}

void ForwardPlusRenderer::createFrameBuffersAndTextures()
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
