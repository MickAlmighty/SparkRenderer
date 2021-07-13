#include "SparkRenderer.h"

#include "Camera.h"
#include "CommonUtils.h"
#include "GUI/ImGui/imgui.h"
#include "Lights/DirectionalLight.h"
#include "Lights/LightProbe.h"
#include "RenderingRequest.h"
#include "ResourceLibrary.h"
#include "Spark.h"
#include "Clock.h"
#include "Timer.h"

namespace spark
{
SparkRenderer* SparkRenderer::getInstance()
{
    static SparkRenderer sparkRenderer{};
    return &sparkRenderer;
}

void SparkRenderer::drawGui()
{
    if(ImGui::BeginMenu("SparkRenderer"))
    {
        const std::string menuName = "SSAO";
        if(ImGui::BeginMenu(menuName.c_str()))
        {
            ImGui::Checkbox("SSAO enabled", &ssaoEnable);
            ImGui::DragInt("Samples", &ao.kernelSize, 1, 0, 64);
            ImGui::DragFloat("Radius", &ao.radius, 0.05f, 0.0f);
            ImGui::DragFloat("Bias", &ao.bias, 0.005f);
            ImGui::DragFloat("Power", &ao.power, 0.05f, 0.0f);
            ImGui::EndMenu();
        }

        const std::string menuName2 = "Depth of Field";
        if(ImGui::BeginMenu(menuName2.c_str()))
        {
            ImGui::Checkbox("DOF enabled", &dofEnable);
            ImGui::DragFloat("NearStart", &nearStart, 0.1f, 0.0f);
            ImGui::DragFloat("NearEnd", &nearEnd, 0.1f);
            ImGui::DragFloat("FarStart", &farStart, 0.1f, 0.0f);
            ImGui::DragFloat("FarEnd", &farEnd, 0.1f, 0.0f);
            ImGui::EndMenu();
        }

        const std::string menuName3 = "Light Shafts";
        if(ImGui::BeginMenu(menuName3.c_str()))
        {
            ImGui::Checkbox("Light shafts enabled", &isLightShaftsPassEnabled);
            ImGui::DragFloat("Exposure", &lightShaftsPass.exposure, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("Decay", &lightShaftsPass.decay, 0.01f, 0.0001f);
            ImGui::DragFloat("Density", &lightShaftsPass.density, 0.01f, 0.0001f, 1.0f);
            ImGui::DragFloat("Weight", &lightShaftsPass.weight, 0.01f, 0.0001f);
            ImGui::EndMenu();
        }

        const std::string menuName4 = "Tone Mapping";
        if(ImGui::BeginMenu(menuName4.c_str()))
        {
            ImGui::DragFloat("minLogLuminance", &toneMapper.minLogLuminance, 0.01f);
            ImGui::DragFloat("logLuminanceRange", &toneMapper.logLuminanceRange, 0.01f);
            ImGui::DragFloat("tau", &toneMapper.tau, 0.01f, 0.0f);
            ImGui::EndMenu();
        }

        const std::string menuName5 = "Bloom";
        if(ImGui::BeginMenu(menuName5.c_str()))
        {
            ImGui::Checkbox("Bloom", &isBloomEnabled);
            ImGui::DragFloat("Intensity", &bloomPass.intensity, 0.1f, 0);
            ImGui::EndMenu();
        }

        const std::string menuName6 = "MotionBlur";
        if(ImGui::BeginMenu(menuName6.c_str()))
        {
            ImGui::Checkbox("Motion Blur", &motionBlurEnable);
            ImGui::EndMenu();
        }

        ImGui::EndMenu();
    }
}

void SparkRenderer::setup(unsigned int windowWidth, unsigned int windowHeight)
{
    width = windowWidth;
    height = windowHeight;

    ao.setup(width, height, cameraUBO);
    toneMapper.setup(width, height);
    bloomPass.setup(width, height);
    dofPass.setup(width, height, cameraUBO);
    lightShaftsPass.setup(width, height);

    cubemapViewMatrices.resizeBuffer(sizeof(glm::mat4) * 6);
    cubemapViewMatrices.updateData(utils::getCubemapViewMatrices(glm::vec3(0)));
    brdfLookupTexture = utils::createBrdfLookupTexture(1024);

    initMembers();
    createFrameBuffersAndTextures();
    updateBufferBindings();
}

void SparkRenderer::initMembers()
{
    screenQuad.setup();
    mainShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("default.glsl");
    screenShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("screen.glsl");
    lightShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("light.glsl");
    motionBlurShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("motionBlur.glsl");
    cubemapShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("cubemap.glsl");
    solidColorShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("solidColor.glsl");
    fxaaShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("fxaa.glsl");
    tileBasedLightCullingShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("tileBasedLightCulling.glsl");
    tileBasedLightingShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("tileBasedLighting.glsl");
    localLightProbesLightingShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("localLightProbesLighting.glsl");
    equirectangularToCubemapShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("equirectangularToCubemap.glsl");
    irradianceShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("irradiance.glsl");
    prefilterShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("prefilter.glsl");
    resampleCubemapShader = Spark::resourceLibrary.getResourceByName<resources::Shader>("resampleCubemap.glsl");
}

void SparkRenderer::updateBufferBindings() const
{
    mainShader->bindUniformBuffer("Camera", cameraUBO);
    lightShader->bindUniformBuffer("Camera", cameraUBO);
    motionBlurShader->bindUniformBuffer("Camera", cameraUBO);
    cubemapShader->bindUniformBuffer("Camera", cameraUBO);
    solidColorShader->bindUniformBuffer("Camera", cameraUBO);

    tileBasedLightCullingShader->bindUniformBuffer("Camera", cameraUBO);

    tileBasedLightCullingShader->bindSSBO("PointLightIndices", pointLightIndices);
    tileBasedLightCullingShader->bindSSBO("SpotLightIndices", spotLightIndices);
    tileBasedLightCullingShader->bindSSBO("LightProbeIndices", lightProbeIndices);

    tileBasedLightingShader->bindUniformBuffer("Camera", cameraUBO);

    tileBasedLightingShader->bindSSBO("PointLightIndices", pointLightIndices);
    tileBasedLightingShader->bindSSBO("SpotLightIndices", spotLightIndices);
    tileBasedLightingShader->bindSSBO("LightProbeIndices", lightProbeIndices);

    localLightProbesLightingShader->bindUniformBuffer("Camera", cameraUBO);

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
}

void SparkRenderer::updateLightBuffersBindings() const
{
    const auto& lightManager = scene->lightManager;
    lightShader->bindSSBO("DirLightData", lightManager->getDirLightSSBO());
    lightShader->bindSSBO("PointLightData", lightManager->getPointLightSSBO());
    lightShader->bindSSBO("SpotLightData", lightManager->getSpotLightSSBO());

    tileBasedLightCullingShader->bindSSBO("DirLightData", lightManager->getDirLightSSBO());
    tileBasedLightCullingShader->bindSSBO("PointLightData", lightManager->getPointLightSSBO());
    tileBasedLightCullingShader->bindSSBO("SpotLightData", lightManager->getSpotLightSSBO());
    tileBasedLightCullingShader->bindSSBO("LightProbeData", lightManager->getLightProbeSSBO());

    tileBasedLightingShader->bindSSBO("DirLightData", lightManager->getDirLightSSBO());
    tileBasedLightingShader->bindSSBO("PointLightData", lightManager->getPointLightSSBO());
    tileBasedLightingShader->bindSSBO("SpotLightData", lightManager->getSpotLightSSBO());
    tileBasedLightingShader->bindSSBO("LightProbeData", lightManager->getLightProbeSSBO());
    localLightProbesLightingShader->bindSSBO("DirLightData", lightManager->getDirLightSSBO());
    localLightProbesLightingShader->bindSSBO("PointLightData", lightManager->getPointLightSSBO());
    localLightProbesLightingShader->bindSSBO("SpotLightData", lightManager->getSpotLightSSBO());
}

void SparkRenderer::renderPass(unsigned int windowWidth, unsigned int windowHeight)
{
    resizeWindowIfNecessary(windowWidth, windowHeight);

    {
        const auto camera = scene->getCamera();
        // if(camera->isDirty())
        {
            updateCameraUBO(camera->getProjectionReversedZInfiniteFarPlane(), camera->getViewMatrix(), camera->getPosition());
            camera->cleanDirty();
        }
    }

    lightProbesRenderPass();

    fillGBuffer(gBuffer);
    textureHandle = ao.process(ssaoEnable, screenQuad, gBuffer);
    // renderLights(lightFrameBuffer, gBuffer);
    tileBasedLightRendering(gBuffer);
    renderCubemap(cubemapFramebuffer);
    helperShapes();
    bloom();
    depthOfField();
    lightShafts();
    motionBlur();
    toneMapping();
    fxaa();
    renderToScreen();
    glDepthFunc(GL_LESS);

    clearRenderQueues();
}

void SparkRenderer::addRenderingRequest(const RenderingRequest& request)
{
    renderQueue[request.shaderType].push_back(request);
}

void SparkRenderer::setScene(const std::shared_ptr<Scene>& scene_)
{
    scene = scene_;
    updateLightBuffersBindings();
}

void SparkRenderer::setCubemap(const std::shared_ptr<PbrCubemapTexture>& cubemap)
{
    pbrCubemap = cubemap;
}

void SparkRenderer::resizeWindowIfNecessary(unsigned int windowWidth, unsigned int windowHeight)
{
    if(width != windowWidth || height != windowHeight)
    {
        if(windowWidth != 0 && windowHeight != 0)
        {
            width = windowWidth;
            height = windowHeight;
            createFrameBuffersAndTextures();
        }
    }
    glViewport(0, 0, width, height);
}

void SparkRenderer::fillGBuffer(const GBuffer& geometryBuffer)
{
    PUSH_DEBUG_GROUP(RENDER_TO_MAIN_FRAMEBUFFER);

    glBindFramebuffer(GL_FRAMEBUFFER, geometryBuffer.framebuffer);
    glClearColor(0, 0, 0, 0);
    glClearDepth(0.0f);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);

    mainShader->use();
    for(auto& request : renderQueue[ShaderType::DEFAULT_SHADER])
    {
        request.mesh->draw(mainShader, request.model);
    }

    POP_DEBUG_GROUP();
}

void SparkRenderer::fillGBuffer(const GBuffer& geometryBuffer, const std::function<bool(const RenderingRequest& request)>& filter)
{
    PUSH_DEBUG_GROUP(RENDER_TO_MAIN_FRAMEBUFFER_FILTERED);

    glBindFramebuffer(GL_FRAMEBUFFER, geometryBuffer.framebuffer);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(0.0f);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);

    mainShader->use();
    for(auto& request : renderQueue[ShaderType::DEFAULT_SHADER])
    {
        if(filter(request))
        {
            request.mesh->draw(mainShader, request.model);
        }
    }

    POP_DEBUG_GROUP();
}

void SparkRenderer::renderLights(GLuint framebuffer, const GBuffer& geometryBuffer)
{
    PUSH_DEBUG_GROUP(PBR_LIGHT);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    const auto cubemap = pbrCubemap.lock();

    lightShader->use();
    if(cubemap)
    {
        std::array<GLuint, 8> textures{geometryBuffer.depthTexture,
                                       geometryBuffer.colorTexture,
                                       geometryBuffer.normalsTexture,
                                       geometryBuffer.roughnessMetalnessTexture,
                                       cubemap->irradianceCubemap,
                                       cubemap->prefilteredCubemap,
                                       brdfLookupTexture,
                                       textureHandle};  // ssao
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
    }
    else
    {
        std::array<GLuint, 8> textures{geometryBuffer.depthTexture,
                                       geometryBuffer.colorTexture,
                                       geometryBuffer.normalsTexture,
                                       geometryBuffer.roughnessMetalnessTexture,
                                       0,
                                       0,
                                       0,
                                       textureHandle};  // ssao
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
    }

    textureHandle = lightingTexture;

    POP_DEBUG_GROUP();
}

void SparkRenderer::tileBasedLightCulling(const GBuffer& geometryBuffer) const
{
    PUSH_DEBUG_GROUP(TILE_BASED_LIGHTS_CULLING);
    pointLightIndices.clearData();
    spotLightIndices.clearData();
    lightProbeIndices.clearData();

    tileBasedLightCullingShader->use();

    glBindTextureUnit(0, geometryBuffer.depthTexture);

    // debug of light count per tile
    /*uint8_t clear[]{0,0,0,0};
    glClearTexImage(lightsPerTileTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, &clear);*/
    glBindImageTexture(5, lightsPerTileTexture, 0, false, 0, GL_READ_WRITE, GL_RGBA16F);

    tileBasedLightCullingShader->dispatchCompute(width / 16, height / 16, 1);

    POP_DEBUG_GROUP();
}

void SparkRenderer::tileBasedLightRendering(const GBuffer& geometryBuffer)
{
    tileBasedLightCulling(geometryBuffer);

    PUSH_DEBUG_GROUP(TILE_BASED_DEFERRED)
    float clearRgba[] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearTexImage(lightingTexture, 0, GL_RGBA, GL_FLOAT, &clearRgba);
    glClearTexImage(brightPassTexture, 0, GL_RGBA, GL_FLOAT, &clearRgba);

    const auto cubemap = pbrCubemap.lock();

    tileBasedLightingShader->use();

    // depth texture as sampler2D
    glBindTextureUnit(0, geometryBuffer.depthTexture);
    if(cubemap)
    {
        glBindTextureUnit(1, cubemap->irradianceCubemap);
        glBindTextureUnit(2, cubemap->prefilteredCubemap);
    }
    glBindTextureUnit(3, brdfLookupTexture);
    glBindTextureUnit(4, textureHandle);

    // textures as images
    glBindImageTexture(0, geometryBuffer.colorTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA8);
    glBindImageTexture(1, geometryBuffer.normalsTexture, 0, false, 0, GL_READ_ONLY, GL_RG16F);
    glBindImageTexture(2, geometryBuffer.roughnessMetalnessTexture, 0, false, 0, GL_READ_ONLY, GL_RG8);

    // output image
    glBindImageTexture(3, lightingTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
    glBindImageTexture(4, brightPassTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);

    tileBasedLightingShader->dispatchCompute(width / 16, height / 16, 1);
    glBindTextures(0, 0, nullptr);

    textureHandle = lightingTexture;

    POP_DEBUG_GROUP();
}

void SparkRenderer::bloom()
{
    if (isBloomEnabled)
    {
        textureHandle = bloomPass.process(screenQuad, lightingTexture, brightPassTexture);
    }
}

void SparkRenderer::lightShafts()
{
    if (isLightShaftsPassEnabled)
    {
        if (auto outputOpt =
            lightShaftsPass.process(scene->getCamera(), gBuffer.depthTexture, textureHandle);
            outputOpt.has_value())
        {
            textureHandle = outputOpt.value();
        }
    }
}

void SparkRenderer::renderCubemap(GLuint framebuffer) const
{
    const auto cubemap = pbrCubemap.lock();
    if(!cubemap)
        return;

    PUSH_DEBUG_GROUP(RENDER_CUBEMAP);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GEQUAL);
    cubemapShader->use();

    glBindTextureUnit(0, cubemap->cubemap);
    cube.draw();
    glBindTextures(0, 1, nullptr);
    glDepthFunc(GL_GREATER);
    glDisable(GL_DEPTH_TEST);

    POP_DEBUG_GROUP();
}

void SparkRenderer::helperShapes()
{
    if(renderQueue[ShaderType::SOLID_COLOR_SHADER].empty())
        return;

    PUSH_DEBUG_GROUP(HELPER_SHAPES);

    glBindFramebuffer(GL_FRAMEBUFFER, cubemapFramebuffer);

    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

    enableWireframeMode();
    // light texture must be bound here
    solidColorShader->use();

    for(auto& request : renderQueue[ShaderType::SOLID_COLOR_SHADER])
    {
        request.mesh->draw(solidColorShader, request.model);
    }

    glEnable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    disableWireframeMode();
    POP_DEBUG_GROUP();
}

void SparkRenderer::depthOfField()
{
    if(!dofEnable)
        return;

    dofPass.setUniforms(nearStart, nearEnd, farStart, farEnd);
    dofPass.render(lightingTexture, gBuffer.depthTexture);
    textureHandle = dofPass.getOutputTexture();
}

void SparkRenderer::toneMapping()
{
    textureHandle = toneMapper.process(textureHandle, screenQuad);
}

void SparkRenderer::motionBlur()
{
    const auto camera = scene->getCamera();
    const glm::mat4 projectionView = camera->getProjectionReversedZInfiniteFarPlane() * camera->getViewMatrix();
    static glm::mat4 prevProjectionView = projectionView;

    if(projectionView == prevProjectionView || !motionBlurEnable)
        return;

    if(static bool initialized = false; !initialized)
    {
        // it is necessary when the scene has been loaded and
        // the difference between current VP and last frame VP matrices generates huge velocities for all pixels
        // so it needs to be reset
        prevProjectionView = projectionView;
        initialized = true;
    }

    PUSH_DEBUG_GROUP(MOTION_BLUR);
    glBindFramebuffer(GL_FRAMEBUFFER, motionBlurFramebuffer);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    motionBlurShader->use();
    motionBlurShader->setMat4("prevViewProj", prevProjectionView);
    motionBlurShader->setFloat("currentFPS", static_cast<float>(Clock::getFPS()));
    const glm::vec2 texelSize = {1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height)};
    motionBlurShader->setVec2("texelSize", texelSize);
    std::array<GLuint, 2> textures2{textureHandle, gBuffer.depthTexture};
    glBindTextures(0, static_cast<GLsizei>(textures2.size()), textures2.data());
    screenQuad.draw();
    glBindTextures(0, static_cast<GLsizei>(textures2.size()), nullptr);

    prevProjectionView = projectionView;
    textureHandle = motionBlurTexture;
    POP_DEBUG_GROUP();
}

void SparkRenderer::fxaa()
{
    PUSH_DEBUG_GROUP(FXAA);

    glBindFramebuffer(GL_FRAMEBUFFER, fxaaFramebuffer);

    fxaaShader->use();
    fxaaShader->setVec2("inversedScreenSize", {1.0f / static_cast<float>(width), 1.0f / static_cast<float>(height)});

    glBindTextureUnit(0, textureHandle);
    screenQuad.draw();
    glBindTextures(0, 2, nullptr);

    textureHandle = fxaaTexture;
    POP_DEBUG_GROUP();
}

void SparkRenderer::renderToScreen() const
{
    PUSH_DEBUG_GROUP(RENDER_TO_SCREEN);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);

    screenShader->use();

    glBindTextureUnit(0, textureHandle);

    screenQuad.draw();

    glEnable(GL_DEPTH_TEST);
    POP_DEBUG_GROUP();
}

void SparkRenderer::clearRenderQueues()
{
    for(auto& [shaderType, shaderRenderList] : renderQueue)
    {
        shaderRenderList.clear();
    }
}

void SparkRenderer::createFrameBuffersAndTextures()
{
    dofPass.createFrameBuffersAndTextures(width, height);
    ao.createFrameBuffersAndTextures(width, height);
    toneMapper.createFrameBuffersAndTextures(width, height);
    bloomPass.createFrameBuffersAndTextures(width, height);
    lightShaftsPass.createFrameBuffersAndTextures(width, height);

    pointLightIndices.resizeBuffer(256 * (uint32_t)glm::ceil(height / 16.0f) * (uint32_t)glm::ceil(width / 16.0f) * sizeof(uint32_t));
    spotLightIndices.resizeBuffer(256 * (uint32_t)glm::ceil(height / 16.0f) * (uint32_t)glm::ceil(width / 16.0f) * sizeof(uint32_t));
    lightProbeIndices.resizeBuffer(256 * (uint32_t)glm::ceil(height / 16.0f) * (uint32_t)glm::ceil(width / 16.0f) * sizeof(uint32_t));

    gBuffer.setup(width, height);

    utils::recreateTexture2D(lightingTexture, width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(brightPassTexture, width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(motionBlurTexture, width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(fxaaTexture, width, height, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(lightsPerTileTexture, width / 16, height / 16, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

    utils::recreateFramebuffer(lightFrameBuffer, {lightingTexture, brightPassTexture});
    utils::recreateFramebuffer(cubemapFramebuffer, {lightingTexture});
    utils::bindDepthTexture(cubemapFramebuffer, gBuffer.depthTexture);
    utils::recreateFramebuffer(motionBlurFramebuffer, {motionBlurTexture});
    utils::recreateFramebuffer(fxaaFramebuffer, {fxaaTexture});
}

void SparkRenderer::cleanup()
{
    deleteFrameBuffersAndTextures();
    ao.cleanup();
    toneMapper.cleanup();
    bloomPass.cleanup();
    lightShaftsPass.cleanup();
}

void SparkRenderer::deleteFrameBuffersAndTextures()
{
    gBuffer.cleanup();

    GLuint textures[6] = {lightingTexture, brightPassTexture, motionBlurTexture, lightsPerTileTexture, fxaaTexture, brdfLookupTexture};
    glDeleteTextures(6, textures);

    GLuint frameBuffers[4] = {lightFrameBuffer, cubemapFramebuffer, motionBlurFramebuffer, fxaaFramebuffer};

    glDeleteFramebuffers(4, frameBuffers);
}

void SparkRenderer::enableWireframeMode()
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}

void SparkRenderer::disableWireframeMode()
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

void SparkRenderer::updateCameraUBO(glm::mat4 projection, glm::mat4 view, glm::vec3 pos)
{
    struct CamData
    {
        glm::vec4 camPos;
        glm::mat4 matrices[4];
    };
    const glm::mat4 invertedView = glm::inverse(view);
    const glm::mat4 invertedProj = glm::inverse(projection);
    const CamData camData{glm::vec4(pos, 1.0f), {view, projection, invertedView, invertedProj}};
    cameraUBO.updateData<CamData>({camData});
}

bool SparkRenderer::checkIfSkyboxChanged() const
{
    static GLuint cubemapId{0};
    if(pbrCubemap.lock())
    {
        if(cubemapId != pbrCubemap.lock()->cubemap)
        {
            cubemapId = pbrCubemap.lock()->cubemap;
            return true;
        }
    }
    else if(cubemapId > 0 && pbrCubemap.lock() == nullptr)
    {
        cubemapId = 0;
        return true;
    }

    return false;
}

void SparkRenderer::lightProbesRenderPass()
{
    const auto& lightProbes = scene->lightManager->getLightProbes();
    const bool renderingNeeded = std::any_of(lightProbes.cbegin(), lightProbes.cend(), [](LightProbe* lp) { return lp->generateLightProbe; });

    const bool lightProbesRebuildNeeded = checkIfSkyboxChanged();

    if(const auto skipRendering = !renderingNeeded && !lightProbesRebuildNeeded; skipRendering)
        return;

    auto t = Timer("Local Light Probe creation");
    utils::createCubemap(lightProbeSceneCubemap, sceneCubemapSize, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, true);
    localLightProbeGBuffer.setup(sceneCubemapSize, sceneCubemapSize);

    utils::createFramebuffer(lightProbeLightFbo, {});
    utils::createFramebuffer(lightProbeSkyboxFbo, {});
    utils::bindDepthTexture(lightProbeSkyboxFbo, localLightProbeGBuffer.depthTexture);

    for(const auto& lightProbe : lightProbes)
    {
        if(const auto skipRendering = !renderingNeeded && !lightProbesRebuildNeeded; skipRendering)
            continue;

        generateLightProbe(lightProbe);
    }

    localLightProbeGBuffer.cleanup();
    glDeleteTextures(1, &lightProbeSceneCubemap);
    glDeleteFramebuffers(1, &lightProbeLightFbo);
    glDeleteFramebuffers(1, &lightProbeSkyboxFbo);
}

void SparkRenderer::generateLightProbe(LightProbe* lightProbe)
{
    PUSH_DEBUG_GROUP(SCENE_TO_CUBEMAP);

    const glm::vec3 localLightProbePosition = lightProbe->getGameObject()->transform.world.getPosition();

    const auto projection = utils::getProjectionReversedZ(sceneCubemapSize, sceneCubemapSize, 90.0f, 0.05f, 100.0f);
    const std::array<glm::mat4, 6> viewMatrices = utils::getCubemapViewMatrices(localLightProbePosition);

    glViewport(0, 0, sceneCubemapSize, sceneCubemapSize);
    for(int i = 0; i < 6; ++i)
    {
        PUSH_DEBUG_GROUP(RENDER_TO_CUBEMAP_FACE)

        updateCameraUBO(projection, viewMatrices[i], localLightProbePosition);

        glBindFramebuffer(GL_FRAMEBUFFER, lightProbeLightFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, lightProbeSceneCubemap, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, lightProbeSkyboxFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, lightProbeSceneCubemap, 0);

        renderSceneToCubemap(localLightProbeGBuffer, lightProbeLightFbo, lightProbeSkyboxFbo);

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
    glViewport(0, 0, width, height);
    POP_DEBUG_GROUP();
}

void SparkRenderer::renderSceneToCubemap(const GBuffer& geometryBuffer, GLuint lightFbo, GLuint skyboxFbo)
{
    fillGBuffer(geometryBuffer, [](const RenderingRequest& request) { return request.gameObject->isStatic() == true; });

    {
        PUSH_DEBUG_GROUP(PBR_LIGHT);

        glBindFramebuffer(GL_FRAMEBUFFER, lightFbo);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);

        localLightProbesLightingShader->use();
        std::array<GLuint, 4> textures{
            geometryBuffer.depthTexture,
            geometryBuffer.colorTexture,
            geometryBuffer.normalsTexture,
            geometryBuffer.roughnessMetalnessTexture,
        };
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);

        POP_DEBUG_GROUP();
    }

    renderCubemap(skyboxFbo);
}
}  // namespace spark
