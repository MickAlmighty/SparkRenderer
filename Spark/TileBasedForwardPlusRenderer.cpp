#include "TileBasedForwardPlusRenderer.hpp"

#include "CommonUtils.h"
#include "Spark.h"

namespace spark
{
TileBasedForwardPlusRenderer::TileBasedForwardPlusRenderer(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                                         const std::shared_ptr<lights::LightManager>& lightManager)
    : Renderer(width, height, cameraUbo), lightCullingPass(width, height, cameraUbo, lightManager)
{
    brdfLookupTexture = utils::createBrdfLookupTexture(1024);

    depthOnlyShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("depthOnly.glsl");
    depthAndNormalsShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("depthAndNormals.glsl");
    lightingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("tileBasedForwardPlusPbrLighting.glsl");

    depthOnlyShader->bindUniformBuffer("Camera", cameraUbo);
    depthAndNormalsShader->bindUniformBuffer("Camera", cameraUbo);
    lightingShader->bindUniformBuffer("Camera", cameraUbo);

    bindLightBuffers(lightManager);
    createFrameBuffersAndTextures();
}

TileBasedForwardPlusRenderer::~TileBasedForwardPlusRenderer()
{
    glDeleteTextures(1, &lightingTexture);
    glDeleteTextures(1, &normalsTexture);
    glDeleteTextures(1, &depthTexture);
    glDeleteTextures(1, &brdfLookupTexture);
    glDeleteFramebuffers(1, &lightingFramebuffer);
}

void TileBasedForwardPlusRenderer::depthPrepass(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const UniformBuffer& cameraUbo)
{
    PUSH_DEBUG_GROUP(DEPTH_PREPASS)
    glBindFramebuffer(GL_FRAMEBUFFER, depthPrepassFramebuffer);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glClearDepth(0.0);
    POP_DEBUG_GROUP()

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
    shader->bindUniformBuffer("Camera", cameraUbo);
    for(auto& request : renderQueue[ShaderType::PBR])
    {
        request.mesh->draw(shader, request.model);
    }
}

GLuint TileBasedForwardPlusRenderer::aoPass()
{
    if(isAmbientOcclusionEnabled)
    {
        return ao.process(depthTexture, normalsTexture);
    }
    return 0;
}

void TileBasedForwardPlusRenderer::lightingPass(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue,
                                       const std::weak_ptr<PbrCubemapTexture>& pbrCubemap, const UniformBuffer& cameraUbo, const GLuint ssaoTexture)
{
    PUSH_DEBUG_GROUP(PBR_LIGHT)
    glBindFramebuffer(GL_FRAMEBUFFER, lightingFramebuffer);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glDepthFunc(GL_EQUAL);

    if(!pbrCubemap.expired())
    {
        glBindTextureUnit(7, pbrCubemap.lock()->irradianceCubemap);
        glBindTextureUnit(8, pbrCubemap.lock()->prefilteredCubemap);
    }
    glBindTextureUnit(9, brdfLookupTexture);
    glBindTextureUnit(10, ssaoTexture);

    lightingShader->use();
    lightingShader->bindUniformBuffer("Camera", cameraUbo);
    lightingShader->setUVec2("viewportSize", {w, h});
    for(auto& request : renderQueue[ShaderType::PBR])
    {
        request.mesh->draw(lightingShader, request.model);
    }

    glDepthFunc(GL_GREATER);
    POP_DEBUG_GROUP()
}

GLuint TileBasedForwardPlusRenderer::process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue,
                                    const std::weak_ptr<PbrCubemapTexture>& pbrCubemap, const UniformBuffer& cameraUbo)
{
    depthPrepass(renderQueue, cameraUbo);
    const GLuint ssaoTexture = aoPass();
    lightCullingPass.process(depthTexture);
    lightingPass(renderQueue, pbrCubemap, cameraUbo, ssaoTexture);
    return lightingTexture;
}

void TileBasedForwardPlusRenderer::bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager)
{
    lightingShader->bindSSBO("DirLightData", lightManager->getDirLightSSBO());
    lightingShader->bindSSBO("PointLightData", lightManager->getPointLightSSBO());
    lightingShader->bindSSBO("SpotLightData", lightManager->getSpotLightSSBO());
    lightingShader->bindSSBO("LightProbeData", lightManager->getLightProbeSSBO());
    lightingShader->bindSSBO("PointLightIndices", lightCullingPass.pointLightIndices);
    lightingShader->bindSSBO("SpotLightIndices", lightCullingPass.spotLightIndices);
    lightingShader->bindSSBO("LightProbeIndices", lightCullingPass.lightProbeIndices);
    lightCullingPass.bindLightBuffers(lightManager);
}

void TileBasedForwardPlusRenderer::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    ao.resize(w, h);
    lightCullingPass.resize(w, h);
    createFrameBuffersAndTextures();
}

GLuint TileBasedForwardPlusRenderer::getDepthTexture() const
{
    return depthTexture;
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
}  // namespace spark