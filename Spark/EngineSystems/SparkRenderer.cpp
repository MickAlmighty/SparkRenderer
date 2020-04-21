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
    utils::createTexture2D(averageLuminance, 1, 1, GL_R16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

    luminanceHistogram.genBuffer(256 * sizeof(uint32_t));
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
    downscaleShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("downScale.glsl");

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
}

void SparkRenderer::renderPass()
{
    resizeWindowIfNecessary();

    {
        const auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();
        if(camera->isDirty())
        {
            struct CamData
            {
                glm::vec4 camPos;
                glm::mat4 matrices[4];
            };
            const glm::mat4 view = camera->getViewMatrix();
            const glm::mat4 projection = camera->getProjectionReversedZInfiniteFarPlane();
            const glm::mat4 invertedView = glm::inverse(view);
            const glm::mat4 invertedProj = glm::inverse(projection);
            const CamData camData{glm::vec4(camera->getPosition(), 1.0f), {view, projection, invertedView, invertedProj}};
            cameraUBO.updateData<CamData>({camData});
            camera->cleanDirty();
        }
    }
    fillGBuffer();
    ssaoComputing();
    renderLights();
    renderCubemap();
    helperShapes();
    bloom();
    depthOfField();
    lightShafts();
    motionBlur();
    toneMapping();
    fxaa();

    renderToScreen();

    PUSH_DEBUG_GROUP(GUI);
    glDepthFunc(GL_LESS);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    POP_DEBUG_GROUP();
    glfwSwapBuffers(Spark::window);

    clearRenderQueues();
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

void SparkRenderer::fillGBuffer()
{
    PUSH_DEBUG_GROUP(RENDER_TO_MAIN_FRAMEBUFFER);

    glBindFramebuffer(GL_FRAMEBUFFER, mainFramebuffer);
    glClearColor(1, 1, 1, 1);
    glClearDepth(0.0f);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);

    mainShader->use();
    for(auto& drawMesh : renderQueue[ShaderType::DEFAULT_SHADER])
    {
        drawMesh(mainShader);
    }

    POP_DEBUG_GROUP();
}

void SparkRenderer::ssaoComputing()
{
    if(!ssaoEnable)
    {
        textureHandle = ssaoDisabledTexture;
        return;
    }

    PUSH_DEBUG_GROUP(SSAO);

    glBindFramebuffer(GL_FRAMEBUFFER, ssaoFramebuffer);
    glClearColor(1.0f, 1.0f, 1.0f, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    GLuint textures[3] = {depthTexture, normalsTexture, randomNormalsTexture};
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

void SparkRenderer::renderLights()
{
    PUSH_DEBUG_GROUP(PBR_LIGHT);

    glBindFramebuffer(GL_FRAMEBUFFER, lightFrameBuffer);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    const auto cubemap = SceneManager::getInstance()->getCurrentScene()->cubemap;

    glDisable(GL_DEPTH_TEST);
    lightShader->use();
    if(cubemap)
    {
        std::array<GLuint, 8> textures{
            depthTexture,      colorTexture, normalsTexture, roughnessMetalnessTexture, cubemap->irradianceCubemap, cubemap->prefilteredCubemap,
            brdfLookupTexture, textureHandle};
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
    }
    else
    {
        std::array<GLuint, 8> textures{
            depthTexture, colorTexture, normalsTexture, roughnessMetalnessTexture, 0, 0, 0, ssaoBlurPass->getBlurredTexture()};
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
    }

    glEnable(GL_DEPTH_TEST);

    textureHandle = lightColorTexture;

    POP_DEBUG_GROUP();
}

void SparkRenderer::renderCubemap() const
{
    const auto cubemap = SceneManager::getInstance()->getCurrentScene()->cubemap;
    if(!cubemap)
        return;

    PUSH_DEBUG_GROUP(RENDER_CUBEMAP);

    glBindFramebuffer(GL_FRAMEBUFFER, cubemapFramebuffer);

    glDepthFunc(GL_GEQUAL);
    cubemapShader->use();

    glBindTextureUnit(0, cubemap->cubemap);
    cube.draw();
    glBindTextures(0, 1, nullptr);
    glDepthFunc(GL_GREATER);

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

        downscaleShader->use();
        downscaleShader->setBool("downscale", downscale);
        downscaleShader->setFloat("intensity", intensity);
        downscaleShader->setVec2("outputTextureSizeInversion",
                                 glm::vec2(1.0f / static_cast<float>(viewportWidth), 1.0f / static_cast<float>(viewportHeight)));
        glBindTextureUnit(0, texture);
        screenQuad.draw();
    };

    const auto upsampleTexture = [&downsampleTexture](GLuint framebuffer, GLuint texture, GLuint viewportWidth, GLuint viewportHeight,
                                                      float intensity = 1.0f) {
        downsampleTexture(framebuffer, texture, viewportWidth, viewportHeight, false, intensity);
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
    glDisable(GL_CULL_FACE);

    enableWireframeMode();
    // light framebuffer is binded here
    solidColorShader->use();

    for(auto& drawMesh : renderQueue[ShaderType::SOLID_COLOR_SHADER])
    {
        drawMesh(solidColorShader);
    }

    glEnable(GL_CULL_FACE);
    disableWireframeMode();
    POP_DEBUG_GROUP();
}

void SparkRenderer::depthOfField()
{
    if(!dofEnable)
        return;

    dofPass->setUniforms(nearStart, nearEnd, farStart, farEnd);
    dofPass->render(lightColorTexture, depthTexture);
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

    glBindTextureUnit(0, depthTexture);
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

    if(projectionView == prevProjectionView || !motionBlurEnable)
        return;

    PUSH_DEBUG_GROUP(MOTION_BLUR);
    glBindFramebuffer(GL_FRAMEBUFFER, motionBlurFramebuffer);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    motionBlurShader->use();
    motionBlurShader->setMat4("prevViewProj", prevProjectionView);
    motionBlurShader->setFloat("currentFPS", static_cast<float>(Clock::getFPS()));
    const glm::vec2 texelSize = {1.0f / static_cast<float>(Spark::WIDTH), 1.0f / static_cast<float>(Spark::HEIGHT)};
    motionBlurShader->setVec2("texelSize", texelSize);
    std::array<GLuint, 2> textures2{textureHandle, depthTexture};
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
    glDisable(GL_DEPTH_TEST);

    toneMappingShader->use();
    toneMappingShader->setVec2("inversedScreenSize", {1.0f / Spark::WIDTH, 1.0f / Spark::HEIGHT});

    glBindTextureUnit(0, textureHandle);
    glBindTextureUnit(1, averageLuminance);
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

    glBindImageTexture(0, averageLuminance, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R16F);
    averageLuminanceComputeShader->dispatchCompute(1, 1, 1);  // localWorkGroups has dimensions of x = 16, y = 16
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void SparkRenderer::fxaa()
{
    PUSH_DEBUG_GROUP(FXAA);

    glBindFramebuffer(GL_FRAMEBUFFER, fxaaFramebuffer);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glDisable(GL_DEPTH_TEST);

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
    glDisable(GL_DEPTH_TEST);

    screenShader->use();

    glBindTextureUnit(0, textureHandle);

    screenQuad.draw();
    POP_DEBUG_GROUP();
}

void SparkRenderer::clearRenderQueues()
{
    for (auto& [shaderType, shaderRenderList] : renderQueue)
    {
        shaderRenderList.clear();
    }
}

void SparkRenderer::createFrameBuffersAndTextures()
{
    dofPass->recreateWithNewSize(Spark::WIDTH, Spark::HEIGHT);
    ssaoBlurPass->recreateWithNewSize(Spark::WIDTH / 2, Spark::HEIGHT / 2);

    utils::createTexture2D(colorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_NEAREST);
    utils::createTexture2D(normalsTexture, Spark::WIDTH, Spark::HEIGHT, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
    utils::createTexture2D(roughnessMetalnessTexture, Spark::WIDTH, Spark::HEIGHT, GL_RG, GL_RG, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_NEAREST);
    utils::createTexture2D(depthTexture, Spark::WIDTH, Spark::HEIGHT, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT, GL_CLAMP_TO_EDGE,
                           GL_NEAREST);

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

    utils::createFramebuffer(mainFramebuffer, {colorTexture, normalsTexture, roughnessMetalnessTexture});
    utils::bindDepthTexture(mainFramebuffer, depthTexture);
    utils::createFramebuffer(lightFrameBuffer, {lightColorTexture, brightPassTexture});
    utils::bindDepthTexture(lightFrameBuffer, depthTexture);
    utils::createFramebuffer(cubemapFramebuffer, {lightColorTexture});
    utils::bindDepthTexture(cubemapFramebuffer, depthTexture);
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
    deleteFrameBuffersAndTextures();
    glDeleteTextures(1, &randomNormalsTexture);
    glDeleteTextures(1, &ssaoDisabledTexture);
    glDeleteTextures(1, &averageLuminance);
    glDeleteTextures(1, &brdfLookupTexture);
}

void SparkRenderer::deleteFrameBuffersAndTextures() const
{
    GLuint textures[14] = {colorTexture,        normalsTexture,     roughnessMetalnessTexture, lightColorTexture,
                           brightPassTexture,   downsampleTexture2, downsampleTexture4,        downsampleTexture8,
                           downsampleTexture16, bloomTexture,       toneMappingTexture,        motionBlurTexture,
                           depthTexture,        ssaoTexture};
    glDeleteTextures(14, textures);

    GLuint frameBuffers[11] = {mainFramebuffer,        lightFrameBuffer,        cubemapFramebuffer,     toneMappingFramebuffer,
                               motionBlurFramebuffer,  ssaoFramebuffer,         downsampleFramebuffer2, downsampleFramebuffer4,
                               downsampleFramebuffer8, downsampleFramebuffer16, bloomFramebuffer};

    glDeleteFramebuffers(11, frameBuffers);
}

void SparkRenderer::enableWireframeMode()
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}

void SparkRenderer::disableWireframeMode()
{
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
}  // namespace spark
