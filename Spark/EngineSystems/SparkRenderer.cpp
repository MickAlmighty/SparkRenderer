#include "EngineSystems/SparkRenderer.h"

#include <functional>
#include <random>

#include <GUI/ImGui/imgui.h>
#include <glm/gtx/compatibility.hpp>

#include "BlurPass.h"
#include "Camera.h"
#include "CommonUtils.h"
#include "DepthOfFieldPass.h"
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

        ImGui::EndMenu();
    }
}

void SparkRenderer::setup(unsigned int windowWidth, unsigned int windowHeight)
{
    width = windowWidth;
    height = windowHeight;

    const auto generateSsaoSamples = [this] {
        const std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
        std::default_random_engine generator{};
        std::vector<glm::vec4> ssaoKernel;
        ssaoKernel.reserve(64);
        for(unsigned int i = 0; i < 64; ++i)
        {
            glm::vec4 sample(randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator), 0.0f);
            sample = glm::normalize(sample);
            sample *= randomFloats(generator);

            float scale = static_cast<float>(i) / 64.0f;
            scale = glm::lerp(0.1f, 1.0f, scale * scale);
            sample *= scale;

            ssaoKernel.push_back(sample);
        }
        sampleUniformBuffer.resizeBuffer(64 * sizeof(glm::vec4));
        sampleUniformBuffer.updateData(ssaoKernel);
    };

    const auto generateSsaoNoiseTexture = [this] {
        const std::uniform_real_distribution<float> randomFloats(0.0, 1.0);
        std::default_random_engine generator{};

        std::vector<glm::vec3> ssaoNoise;
        ssaoNoise.reserve(16);
        for(unsigned int i = 0; i < 16; i++)
        {
            glm::vec3 noise(randomFloats(generator) * 2.0f - 1.0f, randomFloats(generator) * 2.0f - 1.0f, 0.0f);
            ssaoNoise.push_back(noise);
        }

        utils::createTexture2D(randomNormalsTexture, 4, 4, GL_RGB32F, GL_RGB, GL_FLOAT, GL_REPEAT, GL_NEAREST, false, ssaoNoise.data());
    };

    generateSsaoSamples();
    generateSsaoNoiseTexture();

    unsigned char red = 255;
    utils::createTexture2D(ssaoDisabledTexture, 1, 1, GL_RED, GL_RED, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_NEAREST, false, &red);
    utils::createTexture2D(averageLuminanceTexture, 1, 1, GL_R16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

    luminanceHistogram.resizeBuffer(256 * sizeof(uint32_t));
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
    mainShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("default.glsl");
    screenShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("screen.glsl");
    toneMappingShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("toneMapping.glsl");
    lightShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("light.glsl");
    motionBlurShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("motionBlur.glsl");
    cubemapShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("cubemap.glsl");
    ssaoShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("ssao.glsl");
    circleOfConfusionShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("circleOfConfusion.glsl");
    bokehDetectionShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("bokehDetection.glsl");
    blendDofShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("blendDof.glsl");
    solidColorShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("solidColor.glsl");
    lightShaftsShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("lightShafts.glsl");
    luminanceHistogramComputeShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("luminanceHistogramCompute.glsl");
    averageLuminanceComputeShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("averageLuminanceCompute.glsl");
    fxaaShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("fxaa.glsl");
    bloomDownScaleShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("bloomDownScale.glsl");
    bloomUpScaleShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("bloomUpScale.glsl");
    tileBasedLightCullingShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("tileBasedLightCulling.glsl");
    tileBasedLightingShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("tileBasedLighting.glsl");
    localLightProbesLightingShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("localLightProbesLighting.glsl");
    equirectangularToCubemapShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("equirectangularToCubemap.glsl");
    irradianceShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("irradiance.glsl");
    prefilterShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("prefilter.glsl");
    resampleCubemapShader = Spark::resourceLibrary.getResourceByNameWithOptLoad<resources::Shader>("resampleCubemap.glsl");

    dofPass = std::make_unique<DepthOfFieldPass>(width, height);
    ssaoBlurPass = std::make_unique<BlurPass>(width / 2, height / 2);
    upsampleBloomBlurPass2 = std::make_unique<BlurPass>(width / 2, height / 2);
    upsampleBloomBlurPass4 = std::make_unique<BlurPass>(width / 4, height / 4);
    upsampleBloomBlurPass8 = std::make_unique<BlurPass>(width / 8, height / 8);
    upsampleBloomBlurPass16 = std::make_unique<BlurPass>(width / 16, height / 16);
}

void SparkRenderer::updateBufferBindings() const
{
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

void SparkRenderer::resizeWindowIfNecessary(unsigned int windowWidth, unsigned int windowHeight)
{
    if(width != static_cast<unsigned int>(windowWidth) || height != static_cast<unsigned int>(windowHeight))
    {
        if(windowWidth != 0 && windowHeight != 0)
        {
            width = windowWidth;
            height = windowHeight;
            deleteFrameBuffersAndTextures();
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

void SparkRenderer::ssaoComputing(const GBuffer& geometryBuffer)
{
    if(!ssaoEnable)
    {
        textureHandle = ssaoDisabledTexture;
        return;
    }

    PUSH_DEBUG_GROUP(SSAO);

    glBindFramebuffer(GL_FRAMEBUFFER, ssaoFramebuffer);
    glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint textures[3] = {geometryBuffer.depthTexture, geometryBuffer.normalsTexture, randomNormalsTexture};
    glBindTextures(0, 3, textures);
    ssaoShader->use();
    ssaoShader->setInt("kernelSize", kernelSize);
    ssaoShader->setFloat("radius", radius);
    ssaoShader->setFloat("bias", bias);
    ssaoShader->setFloat("power", power);
    ssaoShader->setVec2("screenSize", {static_cast<float>(width), static_cast<float>(height)});
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

    const auto cubemap = scene->skybox;

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
    glClearTexImage(lightColorTexture, 0, GL_RGBA, GL_FLOAT, &clearRgba);
    glClearTexImage(brightPassTexture, 0, GL_RGBA, GL_FLOAT, &clearRgba);

    const auto cubemap = scene->skybox;

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

    tileBasedLightingShader->dispatchCompute(width / 16, height / 16, 1);
    glBindTextures(0, 0, nullptr);

    textureHandle = lightColorTexture;

    POP_DEBUG_GROUP();
}

void SparkRenderer::renderCubemap(GLuint framebuffer) const
{
    const auto cubemap = scene->skybox;
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

    upsampleTexture(bloomFramebuffer, lightColorTexture, width, height);

    downsampleTexture(downsampleFramebuffer2, brightPassTexture, width / 2, height / 2);
    downsampleTexture(downsampleFramebuffer4, downsampleTexture2, width / 4, height / 4);
    downsampleTexture(downsampleFramebuffer8, downsampleTexture4, width / 8, height / 8);
    downsampleTexture(downsampleFramebuffer16, downsampleTexture8, width / 16, height / 16);

    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    glEnable(GL_BLEND);

    upsampleBloomBlurPass16->blurTexture(downsampleTexture16);
    upsampleTexture(downsampleFramebuffer8, upsampleBloomBlurPass16->getBlurredTexture(), width / 8, height / 8);

    upsampleBloomBlurPass8->blurTexture(downsampleTexture8);
    upsampleTexture(downsampleFramebuffer4, upsampleBloomBlurPass8->getBlurredTexture(), width / 4, height / 4);

    upsampleBloomBlurPass4->blurTexture(downsampleTexture4);
    upsampleTexture(downsampleFramebuffer2, upsampleBloomBlurPass4->getBlurredTexture(), width / 2, height / 2);

    upsampleBloomBlurPass2->blurTexture(downsampleTexture2);
    upsampleTexture(bloomFramebuffer, upsampleBloomBlurPass2->getBlurredTexture(), width, height, bloomIntensity);

    textureHandle = bloomTexture;

    glDisable(GL_BLEND);

    glViewport(0, 0, width, height);
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
    const auto& dirLights = scene->lightManager->getDirLights();
    if(dirLights.empty() || lightShaftsEnable != true)
        return;

    const auto camera = scene->getCamera();

    const glm::mat4 view = camera->getViewMatrix();
    const glm::mat4 projection = camera->getProjectionReversedZ();

    const glm::vec3 camPos = camera->getPosition();

    glm::vec3 dirLightPosition = dirLights[0]->getDirection() * -glm::vec3(100);

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
    glViewport(0, 0, width / 2, height / 2);
    glBindFramebuffer(GL_FRAMEBUFFER, lightShaftFramebuffer);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glm::mat4 m1(1);
    glm::mat4 m2(1);

    lightShaftsShader->use();
    lightShaftsShader->setVec2("lightScreenPos", lightScreenPos);
    lightShaftsShader->setVec3("lightColor", dirLights[0]->getColor());
    lightShaftsShader->setInt("samples", samples);
    lightShaftsShader->setFloat("exposure", exposure);
    lightShaftsShader->setFloat("decay", decay);
    lightShaftsShader->setFloat("density", density);
    lightShaftsShader->setFloat("weight", weight);

    glBindTextureUnit(0, gBuffer.depthTexture);
    screenQuad.draw();
    glBindTextureUnit(0, 0);

    glViewport(0, 0, width, height);
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);
    ssaoBlurPass->blurTexture(lightShaftTexture);

    textureHandle = ssaoBlurPass->getBlurredTexture();
    POP_DEBUG_GROUP();
}

void SparkRenderer::motionBlur()
{
    const auto camera = scene->getCamera();
    const glm::mat4 projectionView = camera->getProjectionReversedZInfiniteFarPlane() * camera->getViewMatrix();
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

void SparkRenderer::toneMapping()
{
    PUSH_DEBUG_GROUP(TONE_MAPPING);

    calculateAverageLuminance();

    glBindFramebuffer(GL_FRAMEBUFFER, toneMappingFramebuffer);

    toneMappingShader->use();
    toneMappingShader->setVec2("inversedScreenSize", {1.0f / width, 1.0f / height});

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
    luminanceHistogram.clearData();  // resetting histogram buffer

    // first compute dispatch
    luminanceHistogramComputeShader->use();

    luminanceHistogramComputeShader->setIVec2("inputTextureSize", glm::ivec2(width, height));
    luminanceHistogramComputeShader->setFloat("minLogLuminance", minLogLuminance);
    luminanceHistogramComputeShader->setFloat("oneOverLogLuminanceRange", oneOverLogLuminanceRange);

    glBindImageTexture(0, textureHandle, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
    luminanceHistogramComputeShader->dispatchCompute(width / 16, height / 16, 1);  // localWorkGroups has dimensions of x = 16, y = 16
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

    // second compute dispatch
    averageLuminanceComputeShader->use();

    averageLuminanceComputeShader->setUInt("pixelCount", width * height);
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
    dofPass->recreateWithNewSize(width, height);
    ssaoBlurPass->recreateWithNewSize(width / 2, height / 2);

    pointLightIndices.resizeBuffer(256 * (uint32_t)glm::ceil(height / 16.0f) * (uint32_t)glm::ceil(width / 16.0f) * sizeof(uint32_t));
    spotLightIndices.resizeBuffer(256 * (uint32_t)glm::ceil(height / 16.0f) * (uint32_t)glm::ceil(width / 16.0f) * sizeof(uint32_t));
    lightProbeIndices.resizeBuffer(256 * (uint32_t)glm::ceil(height / 16.0f) * (uint32_t)glm::ceil(width / 16.0f) * sizeof(uint32_t));

    gBuffer.setup(width, height);

    utils::createTexture2D(lightColorTexture, width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(brightPassTexture, width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(toneMappingTexture, width, height, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(motionBlurTexture, width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(lightShaftTexture, width / 2, height / 2, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(ssaoTexture, width, height, GL_RED, GL_RED, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(fxaaTexture, width, height, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(downsampleTexture2, width / 2, height / 2, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(downsampleTexture4, width / 4, height / 4, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(downsampleTexture8, width / 8, height / 8, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(downsampleTexture16, width / 16, height / 16, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(bloomTexture, width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture2D(lightsPerTileTexture, width / 16, height / 16, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

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

bool SparkRenderer::checkIfSkyboxChanged() const
{
    static GLuint cubemapId{0};
    if(scene->skybox)
    {
        if(cubemapId != scene->skybox->cubemap)
        {
            cubemapId = scene->skybox->cubemap;
            return true;
        }
    }
    else if(cubemapId > 0 && scene->skybox == nullptr)
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
