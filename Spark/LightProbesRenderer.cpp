#include "LightProbesRenderer.hpp"

#include "CommonUtils.h"
#include "Spark.h"
#include "Timer.h"
#include "lights/LightProbe.h"

namespace spark
{
LightProbesRenderer::LightProbesRenderer(const std::shared_ptr<lights::LightManager>& lightManager)
    : localLightProbeGBuffer(sceneCubemapSize, sceneCubemapSize), skyboxPass(2, 2)
{
    cubemapViewMatrices.resizeBuffer(sizeof(glm::mat4) * 6);
    cubemapViewMatrices.updateData(utils::getCubemapViewMatrices(glm::vec3(0)));

    localLightProbesLightingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("localLightProbesLighting.glsl");
    equirectangularToCubemapShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("equirectangularToCubemap.glsl");
    irradianceShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("irradiance.glsl");
    prefilterShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("prefilter.glsl");
    resampleCubemapShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("resampleCubemap.glsl");

    localLightProbesLightingShader->bindUniformBuffer("Camera", cameraUbo);
    irradianceShader->bindSSBO("Views", cubemapViewMatrices);
    prefilterShader->bindSSBO("Views", cubemapViewMatrices);
    resampleCubemapShader->bindSSBO("Views", cubemapViewMatrices);
    equirectangularToCubemapShader->bindSSBO("Views", cubemapViewMatrices);

    const glm::mat4 cubemapProjection = utils::getProjectionReversedZ(sceneCubemapSize, sceneCubemapSize, 90.0f, 0.05f, 100.0f);
    irradianceShader->use();
    irradianceShader->setMat4("projection", cubemapProjection);
    prefilterShader->use();
    prefilterShader->setMat4("projection", cubemapProjection);
    resampleCubemapShader->use();
    resampleCubemapShader->setMat4("projection", cubemapProjection);
    equirectangularToCubemapShader->use();
    equirectangularToCubemapShader->setMat4("projection", cubemapProjection);

    bindLightBuffers(lightManager);
}

void LightProbesRenderer::process(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const std::shared_ptr<PbrCubemapTexture>& cubemap,
                                  const std::vector<lights::LightProbe*>& lightProbes)
{
    if(lightProbes.empty())
        return;

    const bool renderingNeeded = std::any_of(lightProbes.cbegin(), lightProbes.cend(), [](lights::LightProbe* lp) { return lp->generateLightProbe; });
    const bool lightProbesRebuildNeeded = checkIfSkyboxChanged(cubemap);
    const auto skipRendering = !renderingNeeded && !lightProbesRebuildNeeded;
    if(skipRendering)
        return;

    auto t = Timer("Local Light Probe creation");
    utils::createCubemap(lightProbeSceneCubemap, sceneCubemapSize, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, true);

    utils::createFramebuffer(lightProbeLightFbo, {});
    utils::createFramebuffer(lightProbeSkyboxFbo, {});
    utils::bindDepthTexture(lightProbeSkyboxFbo, localLightProbeGBuffer.depthTexture);

    for(const auto& lightProbe : lightProbes)
    {
        if(!lightProbe->generateLightProbe && skipRendering)
            continue;

        generateLightProbe(renderQueue, lightProbe, cubemap);
    }

    glDeleteTextures(1, &lightProbeSceneCubemap);
    glDeleteFramebuffers(1, &lightProbeLightFbo);
    glDeleteFramebuffers(1, &lightProbeSkyboxFbo);
}

void LightProbesRenderer::bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager)
{
    localLightProbesLightingShader->bindSSBO("DirLightData", lightManager->getDirLightSSBO());
    localLightProbesLightingShader->bindSSBO("PointLightData", lightManager->getPointLightSSBO());
    localLightProbesLightingShader->bindSSBO("SpotLightData", lightManager->getSpotLightSSBO());
}

bool LightProbesRenderer::checkIfSkyboxChanged(const std::shared_ptr<PbrCubemapTexture>& cubemap) const
{
    if(static GLuint cubemapId{0}; cubemap)
    {
        if(cubemapId != cubemap->cubemap)
        {
            cubemapId = cubemap->cubemap;
            return true;
        }
    }
    else if(cubemapId > 0 && cubemap == nullptr)
    {
        cubemapId = 0;
        return true;
    }

    return false;
}

void LightProbesRenderer::generateLightProbe(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, lights::LightProbe* lightProbe,
                                             const std::shared_ptr<PbrCubemapTexture>& cubemap)
{
    PUSH_DEBUG_GROUP(SCENE_TO_CUBEMAP);

    const glm::vec3 localLightProbePosition = lightProbe->getGameObject()->transform.world.getPosition();

    const auto projection = utils::getProjectionReversedZ(sceneCubemapSize, sceneCubemapSize, 90.0f, 0.05f, 100.0f);
    const std::array<glm::mat4, 6> viewMatrices = utils::getCubemapViewMatrices(localLightProbePosition);

    glViewport(0, 0, sceneCubemapSize, sceneCubemapSize);
    for(int i = 0; i < 6; ++i)
    {
        PUSH_DEBUG_GROUP(RENDER_TO_CUBEMAP_FACE)

        utils::updateCameraUBO(cameraUbo, projection, viewMatrices[i], localLightProbePosition);

        glBindFramebuffer(GL_FRAMEBUFFER, lightProbeLightFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, lightProbeSceneCubemap, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, lightProbeSkyboxFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, lightProbeSceneCubemap, 0);

        renderSceneToCubemap(renderQueue, cubemap);

        POP_DEBUG_GROUP();
    }

    // this mipmap chain for all faces is needed for filtered importance sampling in prefilter stage
    glGenerateTextureMipmap(lightProbeSceneCubemap);

    // light probe generation
    GLuint lightProbeFramebuffer{};
    glGenFramebuffers(1, &lightProbeFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, lightProbeFramebuffer);

    lightProbe->renderIntoIrradianceCubemap(lightProbeFramebuffer, lightProbeSceneCubemap, cube, irradianceShader);
    lightProbe->renderIntoPrefilterCubemap(lightProbeFramebuffer, lightProbeSceneCubemap, sceneCubemapSize, cube, prefilterShader,
                                           resampleCubemapShader);
    lightProbe->generateLightProbe = false;

    glDeleteFramebuffers(1, &lightProbeFramebuffer);
    POP_DEBUG_GROUP();
}

void LightProbesRenderer::renderSceneToCubemap(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue,
                                               const std::shared_ptr<PbrCubemapTexture>& cubemap)
{
    constexpr auto filterStaticObjectsOnly = [](const RenderingRequest& request) { return request.gameObject->isStatic() == true; };
    localLightProbeGBuffer.fill(renderQueue, filterStaticObjectsOnly, cameraUbo);

    {
        PUSH_DEBUG_GROUP(PBR_LIGHT);

        glViewport(0, 0, sceneCubemapSize, sceneCubemapSize);
        glBindFramebuffer(GL_FRAMEBUFFER, lightProbeLightFbo);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        localLightProbesLightingShader->use();
        std::array<GLuint, 4> textures{
            localLightProbeGBuffer.depthTexture,
            localLightProbeGBuffer.colorTexture,
            localLightProbeGBuffer.normalsTexture,
            localLightProbeGBuffer.roughnessMetalnessTexture,
        };
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);

        POP_DEBUG_GROUP();
    }

    skyboxPass.processFramebuffer(cubemap, lightProbeSkyboxFbo, sceneCubemapSize, sceneCubemapSize, cameraUbo);
}
}  // namespace spark
