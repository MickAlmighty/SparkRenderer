#include "EngineSystems/SparkRenderer.h"

#include <exception>
#include <functional>
#include <random>

#include <GUI/ImGuizmo.h>
#include <GUI/ImGui/imgui.h>
#include <GUI/ImGui/imgui_impl_glfw.h>
#include <GUI/ImGui/imgui_impl_opengl3.h>
#include <glm/gtx/compatibility.hpp>

#include "BlurPass.h"
#include "Camera.h"
#include "CommonUtils.h"
#include "DepthOfFieldPass.h"
#include "EngineSystems/SceneManager.h"
#include "Lights/LightProbe.h"
#include "RenderingRequest.h"
#include "ResourceLibrary.h"
#include "Scene.h"
#include "Spark.h"
#include "Clock.h"
#include "Logging.h"

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
            ImGui::DragInt("Samples", &kernelSize, 1, 0, 64);
            ImGui::DragFloat("Radius", &radius, 0.05f, 0.0f);
            ImGui::DragFloat("Bias", &bias, 0.005f);
            ImGui::DragFloat("Power", &power, 0.05f, 0.0f);
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
            ImGui::Checkbox("Light shafts enabled", &lightShaftsEnable);
            ImGui::DragInt("Samples", &samples, 1, 0);
            ImGui::DragFloat("Exposure", &exposure, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("Decay", &decay, 0.01f, 1.0f);
            ImGui::DragFloat("Density", &density, 0.01f, 0.0f, 1.0f);
            ImGui::DragFloat("Weight", &weight, 0.01f, 0.0f, 1.0f);
            ImGui::EndMenu();
        }

        const std::string menuName4 = "Tone Mapping";
        if(ImGui::BeginMenu(menuName4.c_str()))
        {
            ImGui::DragFloat("minLogLuminance", &minLogLuminance, 0.01f);
            ImGui::DragFloat("logLuminanceRange", &logLuminanceRange, 0.01f);
            ImGui::DragFloat("tau", &tau, 0.01f, 0.0f);
            ImGui::EndMenu();
        }

        const std::string menuName5 = "Bloom";
        if(ImGui::BeginMenu(menuName5.c_str()))
        {
            ImGui::Checkbox("Bloom", &bloomEnable);
            ImGui::DragFloat("Intensity", &bloomIntensity, 0.1f, 0);
            ImGui::EndMenu();
        }

        const std::string menuName6 = "MotionBlur";
        if(ImGui::BeginMenu(menuName6.c_str()))
        {
            ImGui::Checkbox("Motion Blur", &motionBlurEnable);
            ImGui::EndMenu();
        }
        const std::string menuName7 = "Rendering Scene To Cubemap";
        if(ImGui::BeginMenu(menuName7.c_str()))
        {
            ImGui::Checkbox("scene to cubemap rendering", &renderingToCubemap);
            ImGui::EndMenu();
        }

        ImGui::EndMenu();
    }
}

void SparkRenderer::setup()
{
    cameraUBO.genBuffer();
    const std::uniform_real_distribution<float> randomFloats(0.0, 1.0);  // random floats between 0.0 - 1.0
    std::default_random_engine generator;
    const auto generateSsaoSamples = [this, &randomFloats, &generator] {
        std::vector<glm::vec3> ssaoKernel;
        ssaoKernel.reserve(64);
        for(unsigned int i = 0; i < 64; ++i)
        {
            glm::vec3 sample(randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator));
            sample = glm::normalize(sample);
            sample *= randomFloats(generator);

            float scale = static_cast<float>(i) / 64.0f;
            scale = glm::lerp(0.1f, 1.0f, scale * scale);
            sample *= scale;

            ssaoKernel.push_back(sample);
        }
        sampleUniformBuffer.genBuffer();
        sampleUniformBuffer.updateData(ssaoKernel);
    };

    const auto generateSsaoNoiseTexture = [this, &randomFloats, &generator] {
        std::vector<glm::vec3> ssaoNoise;
        ssaoNoise.reserve(16);
        for(unsigned int i = 0; i < 16; i++)
        {
            glm::vec3 noise(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, 0.0f);
            ssaoNoise.push_back(noise);
        }

        utils::createTexture2D(randomNormalsTexture, 4, 4, GL_RGB32F, GL_RGB, GL_FLOAT, GL_REPEAT, GL_NEAREST, false, ssaoNoise.data());
    };

    generateSsaoSamples();
    generateSsaoNoiseTexture();

    unsigned char red = 255;
    utils::createTexture2D(ssaoDisabledTexture, 1, 1, GL_RED, GL_RED, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_NEAREST, false, &red);
    utils::createTexture2D(averageLuminanceTexture, 1, 1, GL_R16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

    luminanceHistogram.genBuffer(256 * sizeof(uint32_t));
    pointLightIndices.genBuffer(256 * (uint32_t)glm::ceil(Spark::HEIGHT / 16.0f) * (uint32_t)glm::ceil(Spark::WIDTH / 16.0f) * sizeof(uint32_t));
    spotLightIndices.genBuffer(256 * (uint32_t)glm::ceil(Spark::HEIGHT / 16.0f) * (uint32_t)glm::ceil(Spark::WIDTH / 16.0f) * sizeof(uint32_t));
    lightProbeIndices.genBuffer(256 * (uint32_t)glm::ceil(Spark::HEIGHT / 16.0f) * (uint32_t)glm::ceil(Spark::WIDTH / 16.0f) * sizeof(uint32_t));
    brdfLookupTexture = utils::createBrdfLookupTexture(1024);

    initMembers();
}

void SparkRenderer::initMembers()
{
    screenQuad.setup();
    mainShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("default.glsl");
    screenShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("screen.glsl");
    toneMappingShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("toneMapping.glsl");
    lightShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("light.glsl");
    motionBlurShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("motionBlur.glsl");
    cubemapShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("cubemap.glsl");
    ssaoShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("ssao.glsl");
    circleOfConfusionShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("circleOfConfusion.glsl");
    bokehDetectionShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("bokehDetection.glsl");
    blendDofShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("blendDof.glsl");
    solidColorShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("solidColor.glsl");
    lightShaftsShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("lightShafts.glsl");
    luminanceHistogramComputeShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("luminanceHistogramCompute.glsl");
    averageLuminanceComputeShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("averageLuminanceCompute.glsl");
    fxaaShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("fxaa.glsl");
    bloomDownScaleShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("bloomDownScale.glsl");
    bloomUpScaleShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("bloomUpScale.glsl");
    tileBasedLightCullingShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("tileBasedLightCulling.glsl");
    tileBasedLightingShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("tileBasedLighting.glsl");

    dofPass = std::make_unique<DepthOfFieldPass>(Spark::WIDTH, Spark::HEIGHT);
    ssaoBlurPass = std::make_unique<BlurPass>(Spark::WIDTH / 2, Spark::HEIGHT / 2);
    upsampleBloomBlurPass2 = std::make_unique<BlurPass>(Spark::WIDTH / 2, Spark::HEIGHT / 2);
    upsampleBloomBlurPass4 = std::make_unique<BlurPass>(Spark::WIDTH / 4, Spark::HEIGHT / 4);
    upsampleBloomBlurPass8 = std::make_unique<BlurPass>(Spark::WIDTH / 8, Spark::HEIGHT / 8);
    upsampleBloomBlurPass16 = std::make_unique<BlurPass>(Spark::WIDTH / 16, Spark::HEIGHT / 16);

    updateBufferBindings();
    createFrameBuffersAndTextures();
}

void SparkRenderer::updateBufferBindings() const
{
    lightShader->use();
    lightShader->bindSSBO("DirLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->dirLightSSBO);
    lightShader->bindSSBO("PointLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->pointLightSSBO);
    lightShader->bindSSBO("SpotLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->spotLightSSBO);

    mainShader->bindUniformBuffer("Camera", cameraUBO);
    lightShader->bindUniformBuffer("Camera", cameraUBO);
    motionBlurShader->bindUniformBuffer("Camera", cameraUBO);
    cubemapShader->bindUniformBuffer("Camera", cameraUBO);
    ssaoShader->bindUniformBuffer("Camera", cameraUBO);
    ssaoShader->bindUniformBuffer("Samples", sampleUniformBuffer);
    circleOfConfusionShader->bindUniformBuffer("Camera", cameraUBO);
    solidColorShader->bindUniformBuffer("Camera", cameraUBO);

    luminanceHistogramComputeShader->bindSSBO("LuminanceHistogram", luminanceHistogram);
    averageLuminanceComputeShader->bindSSBO("LuminanceHistogram", luminanceHistogram);

    tileBasedLightCullingShader->bindUniformBuffer("Camera", cameraUBO);
    tileBasedLightCullingShader->bindSSBO("DirLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->dirLightSSBO);
    tileBasedLightCullingShader->bindSSBO("PointLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->pointLightSSBO);
    tileBasedLightCullingShader->bindSSBO("SpotLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->spotLightSSBO);
    tileBasedLightCullingShader->bindSSBO("LightProbeData", SceneManager::getInstance()->getCurrentScene()->lightManager->lightProbeSSBO);

    tileBasedLightCullingShader->bindSSBO("PointLightIndices", pointLightIndices);
    tileBasedLightCullingShader->bindSSBO("SpotLightIndices", spotLightIndices);
    tileBasedLightCullingShader->bindSSBO("LightProbeIndices", lightProbeIndices);

    tileBasedLightingShader->bindUniformBuffer("Camera", cameraUBO);
    tileBasedLightingShader->bindSSBO("DirLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->dirLightSSBO);
    tileBasedLightingShader->bindSSBO("PointLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->pointLightSSBO);
    tileBasedLightingShader->bindSSBO("SpotLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->spotLightSSBO);
    tileBasedLightingShader->bindSSBO("LightProbeData", SceneManager::getInstance()->getCurrentScene()->lightManager->lightProbeSSBO);

    tileBasedLightingShader->bindSSBO("PointLightIndices", pointLightIndices);
    tileBasedLightingShader->bindSSBO("SpotLightIndices", spotLightIndices);
    tileBasedLightingShader->bindSSBO("LightProbeIndices", lightProbeIndices);
}

void SparkRenderer::renderPass()
{
    resizeWindowIfNecessary();

    {
        const auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();
        // if(camera->isDirty())
        {
            updateCameraUBO(camera->getProjectionReversedZ(), camera->getViewMatrix(), camera->getPosition());
            camera->cleanDirty();
        }
    }
    fillGBuffer(gBuffer);
    ssaoComputing(gBuffer);
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

    for(const auto& lightProbeWeakPtr : SceneManager::getInstance()->getCurrentScene()->lightManager->lightProbes)
    {
        if(lightProbeWeakPtr.expired())
            continue;

        const auto lightProbe = lightProbeWeakPtr.lock();
        if(lightProbe->generateLightProbe)
        {
            generateLightProbe(lightProbe);
        }
    }

    /*if(renderingToCubemap)
    {
        renderSceneToCubemap();
        renderingToCubemap = !renderingToCubemap;
    }*/

    PUSH_DEBUG_GROUP(GUI);
    glDepthFunc(GL_LESS);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    POP_DEBUG_GROUP();
    glfwSwapBuffers(Spark::window);

    clearRenderQueues();
}

void SparkRenderer::addRenderingRequest(const RenderingRequest& request)
{
    renderQueue[request.shaderType].push_back(request);
}

void SparkRenderer::resizeWindowIfNecessary()
{
    int width, height;
    glfwGetWindowSize(Spark::window, &width, &height);
    if(Spark::WIDTH != static_cast<unsigned int>(width) || Spark::HEIGHT != static_cast<unsigned int>(height))
    {
        if(width != 0 && height != 0)
        {
            Spark::WIDTH = width;
            Spark::HEIGHT = height;
            deleteFrameBuffersAndTextures();
            createFrameBuffersAndTextures();
        }
    }
    glViewport(0, 0, Spark::WIDTH, Spark::HEIGHT);
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

void SparkRenderer::ssaoComputing(const GBuffer& geometryBuffer)
{
    if(!ssaoEnable)
    {
        textureHandle = ssaoDisabledTexture;
        return;
    }

    PUSH_DEBUG_GROUP(SSAO);

    glBindFramebuffer(GL_FRAMEBUFFER, ssaoFramebuffer);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint textures[3] = {geometryBuffer.depthTexture, geometryBuffer.normalsTexture, randomNormalsTexture};
    glBindTextures(0, 3, textures);
    ssaoShader->use();
    ssaoShader->setInt("kernelSize", kernelSize);
    ssaoShader->setFloat("radius", radius);
    ssaoShader->setFloat("bias", bias);
    ssaoShader->setFloat("power", power);
    ssaoShader->setVec2("screenSize", {static_cast<float>(Spark::WIDTH), static_cast<float>(Spark::HEIGHT)});
    // uniforms have default values in shader
    screenQuad.draw();
    glBindTextures(0, 3, nullptr);

    ssaoBlurPass->blurTexture(ssaoTexture);

    textureHandle = ssaoBlurPass->getBlurredTexture();
    POP_DEBUG_GROUP();
}

void SparkRenderer::renderLights(GLuint framebuffer, const GBuffer& geometryBuffer)
{
    PUSH_DEBUG_GROUP(PBR_LIGHT);

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    const auto cubemap = SceneManager::getInstance()->getCurrentScene()->cubemap;

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

    textureHandle = lightColorTexture;

    POP_DEBUG_GROUP();
}

void SparkRenderer::tileBasedLightCulling(const GBuffer& geometryBuffer) const
{
    PUSH_DEBUG_GROUP(TILE_BASED_LIGHTS_CULLING);
    pointLightIndices.clearBuffer();
    spotLightIndices.clearBuffer();
    lightProbeIndices.clearBuffer();

    tileBasedLightCullingShader->use();

    glBindTextureUnit(0, geometryBuffer.depthTexture);

    // debug of light count per tile
    float clear = 0.0f;
    glClearTexImage(lightsPerTileTexture, 0, GL_RED, GL_FLOAT, &clear);
    glBindImageTexture(7, lightsPerTileTexture, 0, false, 0, GL_READ_WRITE, GL_R32F);

    tileBasedLightCullingShader->dispatchCompute(Spark::WIDTH / 16, Spark::HEIGHT / 16, 1);

    POP_DEBUG_GROUP();
}

void SparkRenderer::tileBasedLightRendering(const GBuffer& geometryBuffer)
{
    tileBasedLightCulling(geometryBuffer);

    PUSH_DEBUG_GROUP(TILE_BASED_DEFERRED)
    float clearRgba[] = {0.0f, 0.0f, 0.0f, 0.0f};
    glClearTexImage(lightColorTexture, 0, GL_RGBA, GL_FLOAT, &clearRgba);
    glClearTexImage(brightPassTexture, 0, GL_RGBA, GL_FLOAT, &clearRgba);

    const auto cubemap = SceneManager::getInstance()->getCurrentScene()->cubemap;

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
    glBindImageTexture(3, lightColorTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
    glBindImageTexture(4, brightPassTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);


    tileBasedLightingShader->dispatchCompute(Spark::WIDTH / 16, Spark::HEIGHT / 16, 1);
    glBindTextures(0, 0, nullptr);

    textureHandle = lightColorTexture;

    POP_DEBUG_GROUP();
}

void SparkRenderer::renderCubemap(GLuint framebuffer) const
{
    const auto cubemap = SceneManager::getInstance()->getCurrentScene()->cubemap;
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

void SparkRenderer::bloom()
{
    if(!bloomEnable)
    {
        return;
    }
    PUSH_DEBUG_GROUP(BLOOM)

    const auto downsampleTexture = [this](GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight, bool downscale = true,
                                          float intensity = 1.0f) {
        glViewport(0, 0, viewportWidth, viewportHeight);

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        // glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        // glClear(GL_COLOR_BUFFER_BIT);

        bloomDownScaleShader->use();
        bloomDownScaleShader->setVec2("outputTextureSizeInversion",
                                      glm::vec2(1.0f / static_cast<float>(viewportWidth), 1.0f / static_cast<float>(viewportHeight)));
        glBindTextureUnit(0, texture);
        screenQuad.draw();
    };

    const auto upsampleTexture = [this](GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight, float intensity = 1.0f) {
        glViewport(0, 0, viewportWidth, viewportHeight);

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

        bloomUpScaleShader->use();
        bloomUpScaleShader->setFloat("intensity", intensity);
        glBindTextureUnit(0, texture);
        screenQuad.draw();
    };

    upsampleTexture(bloomFramebuffer, lightColorTexture, Spark::WIDTH, Spark::HEIGHT);

    downsampleTexture(downsampleFramebuffer2, brightPassTexture, Spark::WIDTH / 2, Spark::HEIGHT / 2);
    downsampleTexture(downsampleFramebuffer4, downsampleTexture2, Spark::WIDTH / 4, Spark::HEIGHT / 4);
    downsampleTexture(downsampleFramebuffer8, downsampleTexture4, Spark::WIDTH / 8, Spark::HEIGHT / 8);
    downsampleTexture(downsampleFramebuffer16, downsampleTexture8, Spark::WIDTH / 16, Spark::HEIGHT / 16);

    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    glEnable(GL_BLEND);

    upsampleBloomBlurPass16->blurTexture(downsampleTexture16);
    upsampleTexture(downsampleFramebuffer8, upsampleBloomBlurPass16->getBlurredTexture(), Spark::WIDTH / 8, Spark::HEIGHT / 8);

    upsampleBloomBlurPass8->blurTexture(downsampleTexture8);
    upsampleTexture(downsampleFramebuffer4, upsampleBloomBlurPass8->getBlurredTexture(), Spark::WIDTH / 4, Spark::HEIGHT / 4);

    upsampleBloomBlurPass4->blurTexture(downsampleTexture4);
    upsampleTexture(downsampleFramebuffer2, upsampleBloomBlurPass4->getBlurredTexture(), Spark::WIDTH / 2, Spark::HEIGHT / 2);

    upsampleBloomBlurPass2->blurTexture(downsampleTexture2);
    upsampleTexture(bloomFramebuffer, upsampleBloomBlurPass2->getBlurredTexture(), Spark::WIDTH, Spark::HEIGHT, bloomIntensity);

    textureHandle = bloomTexture;

    glDisable(GL_BLEND);

    glViewport(0, 0, Spark::WIDTH, Spark::HEIGHT);
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

    dofPass->setUniforms(nearStart, nearEnd, farStart, farEnd);
    dofPass->render(lightColorTexture, gBuffer.depthTexture);
    textureHandle = dofPass->getOutputTexture();
}

void SparkRenderer::lightShafts()
{
    const auto& dirLights = SceneManager::getInstance()->getCurrentScene()->lightManager->directionalLights;
    if(dirLights.empty() || lightShaftsEnable != true)
        return;

    const auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();

    const glm::mat4 view = camera->getViewMatrix();
    const glm::mat4 projection = camera->getProjectionReversedZ();

    const glm::vec3 camPos = camera->getPosition();

    glm::vec3 dirLightPosition = dirLights[0].lock()->getDirection() * -glm::vec3(100);

    glm::vec4 dirLightNDCpos = projection * view * glm::vec4(dirLightPosition, 1.0f);
    dirLightNDCpos /= dirLightNDCpos.w;

    glm::vec2 lightScreenPos = glm::vec2((dirLightNDCpos.x + 1.0f) * 0.5f, (dirLightNDCpos.y + 1.0f) * 0.5f);

    glBindFramebuffer(GL_FRAMEBUFFER, ssaoBlurPass->getSecondPassFramebuffer());
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if(lightScreenPos.x < 0.0f || lightScreenPos.x > 1.0f || lightScreenPos.y < 0.0f || lightScreenPos.y > 1.0f)
    {
        return;
    }
    // std::cout << "Light world pos: " << dirLightPosition.x << ", " << dirLightPosition.y << ", " << dirLightPosition.z << std::endl;
    // std::cout << "Light on the screen. Pos: " << lightScreenPos.x << ", "<< lightScreenPos.y<< std::endl;
    PUSH_DEBUG_GROUP(LIGHT SHAFTS);
    glViewport(0, 0, Spark::WIDTH / 2, Spark::HEIGHT / 2);
    glBindFramebuffer(GL_FRAMEBUFFER, lightShaftFramebuffer);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glm::mat4 m1(1);
    glm::mat4 m2(1);

    lightShaftsShader->use();
    lightShaftsShader->setVec2("lightScreenPos", lightScreenPos);
    lightShaftsShader->setVec3("lightColor", dirLights[0].lock()->getColor());
    lightShaftsShader->setInt("samples", samples);
    lightShaftsShader->setFloat("exposure", exposure);
    lightShaftsShader->setFloat("decay", decay);
    lightShaftsShader->setFloat("density", density);
    lightShaftsShader->setFloat("weight", weight);

    glBindTextureUnit(0, gBuffer.depthTexture);
    screenQuad.draw();
    glBindTextureUnit(0, 0);

    glViewport(0, 0, Spark::WIDTH, Spark::HEIGHT);
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);
    ssaoBlurPass->blurTexture(lightShaftTexture);

    textureHandle = ssaoBlurPass->getBlurredTexture();
    POP_DEBUG_GROUP();
}

void SparkRenderer::motionBlur()
{
    const auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();
    const glm::mat4 projectionView = camera->getProjectionReversedZ() * camera->getViewMatrix();
    static glm::mat4 prevProjectionView = projectionView;
    static bool initialized = false;

    if(projectionView == prevProjectionView || !motionBlurEnable)
        return;

    if(!initialized)
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
    const glm::vec2 texelSize = {1.0f / static_cast<float>(Spark::WIDTH), 1.0f / static_cast<float>(Spark::HEIGHT)};
    motionBlurShader->setVec2("texelSize", texelSize);
    std::array<GLuint, 2> textures2{textureHandle, gBuffer.depthTexture};
    glBindTextures(0, static_cast<GLsizei>(textures2.size()), textures2.data());
    screenQuad.draw();
    glBindTextures(0, static_cast<GLsizei>(textures2.size()), nullptr);

    prevProjectionView = projectionView;
    textureHandle = motionBlurTexture;
    POP_DEBUG_GROUP();
}

void SparkRenderer::toneMapping()
{
    PUSH_DEBUG_GROUP(TONE_MAPPING);

    calculateAverageLuminance();

    glBindFramebuffer(GL_FRAMEBUFFER, toneMappingFramebuffer);

    toneMappingShader->use();
    toneMappingShader->setVec2("inversedScreenSize", {1.0f / Spark::WIDTH, 1.0f / Spark::HEIGHT});

    glBindTextureUnit(0, textureHandle);
    glBindTextureUnit(1, averageLuminanceTexture);
    screenQuad.draw();
    glBindTextures(0, 3, nullptr);

    textureHandle = toneMappingTexture;
    POP_DEBUG_GROUP();
}

void SparkRenderer::calculateAverageLuminance()
{
    oneOverLogLuminanceRange = 1.0f / logLuminanceRange;

    // this buffer is attached to both shaders in method SparkRenderer::updateBufferBindings()
    luminanceHistogram.clearBuffer();  // resetting histogram buffer

    // first compute dispatch
    luminanceHistogramComputeShader->use();

    luminanceHistogramComputeShader->setIVec2("inputTextureSize", glm::ivec2(Spark::WIDTH, Spark::HEIGHT));
    luminanceHistogramComputeShader->setFloat("minLogLuminance", minLogLuminance);
    luminanceHistogramComputeShader->setFloat("oneOverLogLuminanceRange", oneOverLogLuminanceRange);

    glBindImageTexture(0, textureHandle, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
    luminanceHistogramComputeShader->dispatchCompute(Spark::WIDTH / 16, Spark::HEIGHT / 16, 1);  // localWorkGroups has dimensions of x = 16, y = 16
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // second compute dispatch
    averageLuminanceComputeShader->use();

    averageLuminanceComputeShader->setUInt("pixelCount", Spark::WIDTH * Spark::HEIGHT);
    averageLuminanceComputeShader->setFloat("minLogLuminance", minLogLuminance);
    averageLuminanceComputeShader->setFloat("logLuminanceRange", logLuminanceRange);
    averageLuminanceComputeShader->setFloat("deltaTime", static_cast<float>(Clock::getDeltaTime()));
    averageLuminanceComputeShader->setFloat("tau", tau);

    glBindImageTexture(0, averageLuminanceTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R16F);
    averageLuminanceComputeShader->dispatchCompute(1, 1, 1);  // localWorkGroups has dimensions of x = 16, y = 16
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void SparkRenderer::fxaa()
{
    PUSH_DEBUG_GROUP(FXAA);

    glBindFramebuffer(GL_FRAMEBUFFER, fxaaFramebuffer);

    fxaaShader->use();
    fxaaShader->setVec2("inversedScreenSize", {1.0f / static_cast<float>(Spark::WIDTH), 1.0f / static_cast<float>(Spark::HEIGHT)});

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
    dofPass->recreateWithNewSize(Spark::WIDTH, Spark::HEIGHT);
    ssaoBlurPass->recreateWithNewSize(Spark::WIDTH / 2, Spark::HEIGHT / 2);

    pointLightIndices.resizeBuffer(256 * (uint32_t)glm::ceil(Spark::HEIGHT / 16.0f) * (uint32_t)glm::ceil(Spark::WIDTH / 16.0f) * sizeof(uint32_t));
    spotLightIndices.resizeBuffer(256 * (uint32_t)glm::ceil(Spark::HEIGHT / 16.0f) * (uint32_t)glm::ceil(Spark::WIDTH / 16.0f) * sizeof(uint32_t));
    lightProbeIndices.resizeBuffer(256 * (uint32_t)glm::ceil(Spark::HEIGHT / 16.0f) * (uint32_t)glm::ceil(Spark::WIDTH / 16.0f) * sizeof(uint32_t));

    gBuffer.setup(Spark::WIDTH, Spark::HEIGHT);

    utils::createTexture2D(lightColorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(brightPassTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(toneMappingTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(motionBlurTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(lightShaftTexture, Spark::WIDTH / 2, Spark::HEIGHT / 2, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(ssaoTexture, Spark::WIDTH, Spark::HEIGHT, GL_RED, GL_RED, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(fxaaTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(downsampleTexture2, Spark::WIDTH / 2, Spark::HEIGHT / 2, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(downsampleTexture4, Spark::WIDTH / 4, Spark::HEIGHT / 4, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(downsampleTexture8, Spark::WIDTH / 8, Spark::HEIGHT / 8, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(downsampleTexture16, Spark::WIDTH / 16, Spark::HEIGHT / 16, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE,
                           GL_LINEAR);
    utils::createTexture2D(bloomTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(lightsPerTileTexture, Spark::WIDTH / 16, Spark::HEIGHT / 16, GL_R32F, GL_RED, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

    utils::createFramebuffer(lightFrameBuffer, {lightColorTexture, brightPassTexture});
    utils::createFramebuffer(cubemapFramebuffer, {lightColorTexture});
    utils::bindDepthTexture(cubemapFramebuffer, gBuffer.depthTexture);
    utils::createFramebuffer(toneMappingFramebuffer, {toneMappingTexture});
    utils::createFramebuffer(motionBlurFramebuffer, {motionBlurTexture});
    utils::createFramebuffer(lightShaftFramebuffer, {lightShaftTexture});
    utils::createFramebuffer(fxaaFramebuffer, {fxaaTexture});
    utils::createFramebuffer(downsampleFramebuffer2, {downsampleTexture2});
    utils::createFramebuffer(downsampleFramebuffer4, {downsampleTexture4});
    utils::createFramebuffer(downsampleFramebuffer8, {downsampleTexture8});
    utils::createFramebuffer(downsampleFramebuffer16, {downsampleTexture16});
    utils::createFramebuffer(bloomFramebuffer, {bloomTexture});

    utils::createFramebuffer(ssaoFramebuffer, {ssaoTexture});
}

void SparkRenderer::cleanup()
{
    cameraUBO.cleanup();
    sampleUniformBuffer.cleanup();
    pointLightIndices.cleanup();
    spotLightIndices.cleanup();
    lightProbeIndices.cleanup();

    deleteFrameBuffersAndTextures();
    glDeleteTextures(1, &randomNormalsTexture);
    glDeleteTextures(1, &ssaoDisabledTexture);
    glDeleteTextures(1, &averageLuminanceTexture);
    glDeleteTextures(1, &brdfLookupTexture);
}

void SparkRenderer::deleteFrameBuffersAndTextures()
{
    gBuffer.cleanup();

    GLuint textures[11] = {lightColorTexture, brightPassTexture,  downsampleTexture2, downsampleTexture4, downsampleTexture8,  downsampleTexture16,
                           bloomTexture,      toneMappingTexture, motionBlurTexture,  ssaoTexture,        lightsPerTileTexture};
    glDeleteTextures(11, textures);

    GLuint frameBuffers[10] = {lightFrameBuffer,       cubemapFramebuffer,     toneMappingFramebuffer, motionBlurFramebuffer,   ssaoFramebuffer,
                               downsampleFramebuffer2, downsampleFramebuffer4, downsampleFramebuffer8, downsampleFramebuffer16, bloomFramebuffer};

    glDeleteFramebuffers(10, frameBuffers);
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

void SparkRenderer::generateLightProbe(const std::shared_ptr<LightProbe>& lightProbe)
{
    PUSH_DEBUG_GROUP(SCENE_TO_CUBEMAP);

    const glm::vec3 viewPos = lightProbe->getGameObject()->transform.world.getPosition();
    const unsigned int cubemapSize = 512;
    const auto projection = utils::getProjectionReversedZ(cubemapSize, cubemapSize, 90.0f, 0.05f, 100.0f);
    // const auto projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    const std::array<glm::mat4, 6> viewMatrices = {glm::lookAt(viewPos, viewPos + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                                                   glm::lookAt(viewPos, viewPos + glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                                                   glm::lookAt(viewPos, viewPos + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
                                                   glm::lookAt(viewPos, viewPos + glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
                                                   glm::lookAt(viewPos, viewPos + glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                                                   glm::lookAt(viewPos, viewPos + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))};

    GLuint sceneCubemap{};
    utils::createCubemap(sceneCubemap, cubemapSize, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    GBuffer geometryBuffer{};
    geometryBuffer.setup(cubemapSize, cubemapSize);

    GLuint lightFbo{}, cubemapFbo{};
    utils::createFramebuffer(lightFbo, {});
    utils::createFramebuffer(cubemapFbo, {});
    utils::bindDepthTexture(cubemapFbo, geometryBuffer.depthTexture);

    glViewport(0, 0, cubemapSize, cubemapSize);
    for(int i = 0; i < 6; ++i)
    {
        PUSH_DEBUG_GROUP(RENDER_TO_FACE)

        updateCameraUBO(projection, viewMatrices[i], viewPos);

        glBindFramebuffer(GL_FRAMEBUFFER, lightFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, sceneCubemap, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, cubemapFbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, sceneCubemap, 0);

        renderSceneToCubemap(geometryBuffer, lightFbo, cubemapFbo);

        POP_DEBUG_GROUP();
    }

    // light probe generation
    const auto irradianceShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("irradiance.glsl");
    const auto prefilterShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("prefilter.glsl");

    const std::array<glm::mat4, 6> cubemapViews = {glm::lookAt(glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                                                   glm::lookAt(glm::vec3(0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                                                   glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
                                                   glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
                                                   glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                                                   glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))};

    GLuint lightProbeFramebuffer{};
    glGenFramebuffers(1, &lightProbeFramebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, lightProbeFramebuffer);

    lightProbe->renderIntoIrradianceCubemap(lightProbeFramebuffer, sceneCubemap, cube, projection, cubemapViews, irradianceShader);
    lightProbe->renderIntoPrefilterCubemap(lightProbeFramebuffer, sceneCubemap, cubemapSize, cube, projection, cubemapViews, prefilterShader);

    geometryBuffer.cleanup();
    glDeleteTextures(1, &sceneCubemap);
    glDeleteFramebuffers(1, &lightFbo);
    glDeleteFramebuffers(1, &cubemapFbo);
    glDeleteFramebuffers(1, &lightProbeFramebuffer);

    glViewport(0, 0, Spark::WIDTH, Spark::HEIGHT);

    lightProbe->generateLightProbe = false;

    POP_DEBUG_GROUP();
}

void SparkRenderer::renderSceneToCubemap(const GBuffer& geometryBuffer, GLuint lightFbo, GLuint skyboxFbo)
{
    fillGBuffer(geometryBuffer, [](const RenderingRequest& request) { return request.gameObject->isStatic() == true; });
    ssaoComputing(geometryBuffer);
    renderLights(lightFbo, geometryBuffer);
    renderCubemap(skyboxFbo);
}
}  // namespace spark
