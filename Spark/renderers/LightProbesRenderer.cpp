#include "LightProbesRenderer.hpp"

#include "utils/CommonUtils.h"
#include "Spark.h"
#include "Timer.h"
#include "lights/LightProbe.h"

namespace spark::renderers
{
LightProbesRenderer::LightProbesRenderer() : localLightProbeGBuffer(sceneCubemapSize, sceneCubemapSize), skyboxPass(2, 2)
{
    cubemapViewMatrices.resizeBuffer(sizeof(glm::mat4) * 6);
    cubemapViewMatrices.updateData(utils::getCubemapViewMatrices(glm::vec3(0)));

    localLightProbesLightingShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/localLightProbesLighting.glsl");
    equirectangularToCubemapShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/equirectangularToCubemap.glsl");
    irradianceShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/irradiance.glsl");
    prefilterShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/prefilter.glsl");
    resampleCubemapShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/resampleCubemap.glsl");

    const glm::mat4 cubemapProjection = utils::getProjectionReversedZ(sceneCubemapSize, sceneCubemapSize, 90.0f, 0.05f, 100.0f);
    irradianceShader->use();
    irradianceShader->setMat4("projection", cubemapProjection);
    prefilterShader->use();
    prefilterShader->setMat4("projection", cubemapProjection);
    resampleCubemapShader->use();
    resampleCubemapShader->setMat4("projection", cubemapProjection);
    equirectangularToCubemapShader->use();
    equirectangularToCubemapShader->setMat4("projection", cubemapProjection);
}

void LightProbesRenderer::process(const std::shared_ptr<Scene>& scene)
{
    const auto& lightProbes = scene->lightManager->getLightProbes();
    if(lightProbes.empty())
        return;

    const bool renderingNeeded = std::any_of(lightProbes.cbegin(), lightProbes.cend(), [](lights::LightProbe* lp) { return lp->generateLightProbe; });
    const bool lightProbesRebuildNeeded = checkIfSkyboxChanged(scene->getSkyboxCubemap().lock());
    const auto skipRendering = !renderingNeeded && !lightProbesRebuildNeeded;
    if(skipRendering)
        return;

    auto t = Timer("Local Light Probe creation");

    localLightProbesLightingShader->bindSSBO("DirLightData", scene->lightManager->getDirLightSSBO());
    localLightProbesLightingShader->bindSSBO("PointLightData", scene->lightManager->getPointLightSSBO());
    localLightProbesLightingShader->bindSSBO("SpotLightData", scene->lightManager->getSpotLightSSBO());

    localLightProbesLightingShader->bindUniformBuffer("Camera", cameraUbo);
    irradianceShader->bindSSBO("Views", cubemapViewMatrices);
    prefilterShader->bindSSBO("Views", cubemapViewMatrices);
    resampleCubemapShader->bindSSBO("Views", cubemapViewMatrices);
    equirectangularToCubemapShader->bindSSBO("Views", cubemapViewMatrices);

    lightProbeSceneCubemap = utils::createCubemap(sceneCubemapSize, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, true);

    utils::createFramebuffer(lightProbeLightFbo, {});
    utils::createFramebuffer(lightProbeSkyboxFbo, {});
    utils::bindDepthTexture(lightProbeSkyboxFbo, localLightProbeGBuffer.depthTexture.get());

    for(const auto& lightProbe : lightProbes)
    {
        if(!lightProbe->generateLightProbe && skipRendering)
            continue;

        generateLightProbe(scene->getRenderingQueues(), lightProbe, scene->getSkyboxCubemap().lock());
    }

    glDeleteFramebuffers(1, &lightProbeLightFbo);
    glDeleteFramebuffers(1, &lightProbeSkyboxFbo);
}

bool LightProbesRenderer::checkIfSkyboxChanged(const std::shared_ptr<PbrCubemapTexture>& cubemap) const
{
    if(static GLuint cubemapId{0}; cubemap)
    {
        if(cubemapId != cubemap->cubemap.get())
        {
            cubemapId = cubemap->cubemap.get();
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

void LightProbesRenderer::generateLightProbe(const std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, lights::LightProbe* lightProbe,
                                             const std::shared_ptr<PbrCubemapTexture>& cubemap)
{
    PUSH_DEBUG_GROUP(SCENE_TO_CUBEMAP);

    const glm::vec3 localLightProbePosition = lightProbe->getGameObject()->transform.world.getPosition();

    const float nearPlane{0.05f};
    const float farPlane{10000.0f};
    const auto projection = utils::getProjectionReversedZ(sceneCubemapSize, sceneCubemapSize, 90.0f, nearPlane, farPlane);
    const std::array<glm::mat4, 6> viewMatrices = utils::getCubemapViewMatrices(localLightProbePosition);

    glViewport(0, 0, sceneCubemapSize, sceneCubemapSize);
    for(int i = 0; i < 6; ++i)
    {
        PUSH_DEBUG_GROUP(RENDER_TO_CUBEMAP_FACE)

        utils::updateCameraUBO(cameraUbo, projection, viewMatrices[i], localLightProbePosition, nearPlane, farPlane);

        glBindFramebuffer(GL_FRAMEBUFFER, lightProbeLightFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, lightProbeSceneCubemap.get(), 0);
        glBindFramebuffer(GL_FRAMEBUFFER, lightProbeSkyboxFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, lightProbeSceneCubemap.get(), 0);

        renderSceneToCubemap(renderQueue, cubemap);

        POP_DEBUG_GROUP();
    }

    // this mipmap chain for all faces is needed for filtered importance sampling in prefilter stage
    glGenerateTextureMipmap(lightProbeSceneCubemap.get());

    // light probe generation
    GLuint lightProbeFramebuffer{};
    glGenFramebuffers(1, &lightProbeFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, lightProbeFramebuffer);

    lightProbe->renderIntoIrradianceCubemap(lightProbeFramebuffer, lightProbeSceneCubemap.get(), cube, irradianceShader);
    lightProbe->renderIntoPrefilterCubemap(lightProbeFramebuffer, lightProbeSceneCubemap.get(), sceneCubemapSize, cube, prefilterShader,
                                           resampleCubemapShader);
    lightProbe->generateLightProbe = false;

    glDeleteFramebuffers(1, &lightProbeFramebuffer);
    POP_DEBUG_GROUP();
}

void LightProbesRenderer::renderSceneToCubemap(const std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue,
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
        const std::array<GLuint, 4> textures{
            localLightProbeGBuffer.depthTexture.get(),
            localLightProbeGBuffer.colorTexture.get(),
            localLightProbeGBuffer.normalsTexture.get(),
            localLightProbeGBuffer.roughnessMetalnessTexture.get(),
        };
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);

        POP_DEBUG_GROUP();
    }

    skyboxPass.processFramebuffer(cubemap, lightProbeSkyboxFbo, sceneCubemapSize, sceneCubemapSize, cameraUbo);
}
}  // namespace spark::renderers
