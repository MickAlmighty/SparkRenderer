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
#include "EngineSystems/ResourceManager.h"
#include "EngineSystems/SceneManager.h"
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
        if (ImGui::BeginMenu(menuName4.c_str()))
        {
            ImGui::DragFloat("minLogLuminance", &minLogLuminance, 0.01f);
            ImGui::DragFloat("logLuminanceRange", &logLuminanceRange, 0.01f);
            ImGui::DragFloat("tau", &tau, 0.01f, 0.0f);
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
        sampleUniformBuffer.update(ssaoKernel);
    };

    const auto generateSsaoNoiseTexture = [this, &randomFloats, &generator] {
        std::vector<glm::vec3> ssaoNoise;
        ssaoNoise.reserve(16);
        for (unsigned int i = 0; i < 16; i++)
        {
            glm::vec3 noise(randomFloats(generator) * 2.0 - 1.0, randomFloats(generator) * 2.0 - 1.0, 0.0f);
            ssaoNoise.push_back(noise);
        }

        glGenTextures(1, &randomNormalsTexture);
        glBindTexture(GL_TEXTURE_2D, randomNormalsTexture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 4, 4, 0, GL_RGB, GL_FLOAT, ssaoNoise.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    };

    generateSsaoSamples();
    generateSsaoNoiseTexture();

    utils::createTexture(ssaoDisabledTexture, 1, 1, GL_RED, GL_RED, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_NEAREST);
    utils::uploadDataToTexture2D(ssaoDisabledTexture, 0, 1, 1, GL_RED, GL_UNSIGNED_BYTE, std::vector<unsigned char>{255});
    utils::createTexture(averageLuminance, 1, 1, GL_R16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

    luminanceHistogram.genBuffer(256 * sizeof(uint32_t));

    initMembers();
}

void SparkRenderer::initMembers()
{
    screenQuad.setup();
    mainShader = ResourceManager::getInstance()->getShader(ShaderType::DEFAULT_SHADER);
    screenShader = ResourceManager::getInstance()->getShader(ShaderType::SCREEN_SHADER);
    toneMappingShader = ResourceManager::getInstance()->getShader(ShaderType::TONE_MAPPING_SHADER);
    lightShader = ResourceManager::getInstance()->getShader(ShaderType::LIGHT_SHADER);
    motionBlurShader = ResourceManager::getInstance()->getShader(ShaderType::MOTION_BLUR_SHADER);
    cubemapShader = ResourceManager::getInstance()->getShader(ShaderType::CUBEMAP_SHADER);
    ssaoShader = ResourceManager::getInstance()->getShader(ShaderType::SSAO_SHADER);
    circleOfConfusionShader = ResourceManager::getInstance()->getShader(ShaderType::COC_SHADER);
    bokehDetectionShader = ResourceManager::getInstance()->getShader(ShaderType::BOKEH_DETECTION_SHADER);
    blendDofShader = ResourceManager::getInstance()->getShader(ShaderType::BLEND_DOF_SHADER);
    solidColorShader = ResourceManager::getInstance()->getShader(ShaderType::SOLID_COLOR_SHADER);

    dofPass = std::make_unique<DepthOfFieldPass>(Spark::WIDTH, Spark::HEIGHT);
    ssaoBlurPass = std::make_unique<BlurPass>(Spark::WIDTH / 2, Spark::HEIGHT / 2);

    updateBufferBindings();
    createFrameBuffersAndTextures();
}

void SparkRenderer::updateBufferBindings() const
{
    const std::shared_ptr<Shader> lShader = lightShader.lock();
    lShader->use();
    lShader->bindSSBO("DirLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->dirLightSSBO);
    lShader->bindSSBO("PointLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->pointLightSSBO);
    lShader->bindSSBO("SpotLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->spotLightSSBO);

    mainShader.lock()->bindUniformBuffer("Camera", cameraUBO);
    lightShader.lock()->bindUniformBuffer("Camera", cameraUBO);
    motionBlurShader.lock()->bindUniformBuffer("Camera", cameraUBO);
    cubemapShader.lock()->bindUniformBuffer("Camera", cameraUBO);
    ssaoShader.lock()->bindUniformBuffer("Camera", cameraUBO);
    ssaoShader.lock()->bindUniformBuffer("Samples", sampleUniformBuffer);
    circleOfConfusionShader.lock()->bindUniformBuffer("Camera", cameraUBO);
    solidColorShader.lock()->bindUniformBuffer("Camera", cameraUBO);

    const auto luminanceHistogramComputeShader = ResourceManager::getInstance()->getShader(ShaderType::LUMINANCE_HISTOGRAM_COMPUTE_SHADER);
    luminanceHistogramComputeShader->bindSSBO("LuminanceHistogram", luminanceHistogram);
    const auto averageLuminanceComputeShader = ResourceManager::getInstance()->getShader(ShaderType::AVERAGE_LUMINANCE_COMPUTE_SHADER);
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
            const glm::mat4 projection = camera->getProjectionReversedZ();
            const glm::mat4 invertedView = glm::inverse(view);
            const glm::mat4 invertedProj = glm::inverse(projection);
            const CamData camData{glm::vec4(camera->getPosition(), 1.0f), {view, projection, invertedView, invertedProj}};
            cameraUBO.update<CamData>({camData});
            camera->cleanDirty();
        }
    }
    fillGBuffer();
    ssaoComputing();
    renderLights();
    renderCubemap();
    helperShapes();
    depthOfField();
    lightShafts();
    fxaa();
    motionBlur();
    toneMapping();

    renderToScreen();

    PUSH_DEBUG_GROUP(GUI);
    glDepthFunc(GL_LESS);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    POP_DEBUG_GROUP();
    glfwSwapBuffers(Spark::window);
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

    glCullFace(GL_BACK);

    std::shared_ptr<Shader> shader = mainShader.lock();
    shader->use();
    for(auto& drawMesh : renderQueue[ShaderType::DEFAULT_SHADER])
    {
        drawMesh(shader);
    }
    renderQueue[ShaderType::DEFAULT_SHADER].clear();

    glCullFace(GL_FRONT);

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
    const auto shader = ssaoShader.lock();
    shader->use();
    shader->setInt("kernelSize", kernelSize);
    shader->setFloat("radius", radius);
    shader->setFloat("bias", bias);
    shader->setFloat("power", power);
    shader->setVec2("screenSize", {static_cast<float>(Spark::WIDTH), static_cast<float>(Spark::HEIGHT)});
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
    const std::shared_ptr<Shader> lShader = lightShader.lock();
    lShader->use();
    if (cubemap)
    {
        std::array<GLuint, 7> textures{ depthTexture,
                                       colorTexture,
                                       normalsTexture,
                                       cubemap->irradianceCubemap,
                                       cubemap->prefilteredCubemap,
                                       cubemap->brdfLUTTexture,
                                       textureHandle };
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
    }
    else
    {
        std::array<GLuint, 7> textures{ depthTexture, colorTexture, normalsTexture, 0, 0, 0, ssaoBlurPass->getBlurredTexture() };
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

    glDepthFunc(GL_GEQUAL);
    const auto cubemapShaderPtr = cubemapShader.lock();
    cubemapShaderPtr->use();

    glBindTextureUnit(0, cubemap->cubemap);
    cube.draw();
    glBindTextures(0, 1, nullptr);
    glDepthFunc(GL_GREATER);

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
    auto solidShader = solidColorShader.lock();
    solidShader->use();

    for(auto& drawMesh : renderQueue[ShaderType::SOLID_COLOR_SHADER])
    {
        drawMesh(solidShader);
    }
    renderQueue[ShaderType::SOLID_COLOR_SHADER].clear();

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

    const auto shader = ResourceManager::getInstance()->getShader(ShaderType::LIGHT_SHAFTS_SHADER);
    shader->use();
    shader->setVec2("lightScreenPos", lightScreenPos);
    shader->setVec3("lightColor", dirLights[0].lock()->getColor());
    shader->setInt("samples", samples);
    shader->setFloat("exposure", exposure);
    shader->setFloat("decay", decay);
    shader->setFloat("density", density);
    shader->setFloat("weight", weight);

    glBindTextureUnit(0, depthTexture);
    screenQuad.draw();
    glBindTextureUnit(0, 0);

    glViewport(0, 0, Spark::WIDTH, Spark::HEIGHT);
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);
    ssaoBlurPass->blurTexture(lightShaftTexture);

    textureHandle = ssaoBlurPass->getBlurredTexture();
    POP_DEBUG_GROUP();
}

void SparkRenderer::fxaa()
{
    PUSH_DEBUG_GROUP(FXAA);

    glBindFramebuffer(GL_FRAMEBUFFER, fxaaFramebuffer);
    glDisable(GL_DEPTH_TEST);

    const auto shader = ResourceManager::getInstance()->getShader(ShaderType::FXAA_SHADER);
    shader->use();
    shader->setVec2("inversedScreenSize", {1.0f / static_cast<float>(Spark::WIDTH), 1.0f / static_cast<float>(Spark::HEIGHT)});

    glBindTextureUnit(0, textureHandle);
    screenQuad.draw();
    glBindTextures(0, 2, nullptr);

    textureHandle = fxaaTexture;
    POP_DEBUG_GROUP();
}

void SparkRenderer::motionBlur()
{
    const auto camera = SceneManager::getInstance()->getCurrentScene()->getCamera();
    const glm::mat4 projectionView = camera->getProjectionReversedZ() * camera->getViewMatrix();
    static glm::mat4 prevProjectionView = projectionView;

    if (projectionView ==prevProjectionView)
        return;

    PUSH_DEBUG_GROUP(MOTION_BLUR);
    const std::shared_ptr<Shader> motionBlurShaderS = motionBlurShader.lock();
    glBindFramebuffer(GL_FRAMEBUFFER, motionBlurFramebuffer);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    motionBlurShaderS->use();
    motionBlurShaderS->setMat4("prevViewProj", prevProjectionView);
    motionBlurShaderS->setFloat("currentFPS", static_cast<float>(Clock::getFPS()));
    const glm::vec2 texelSize = {1.0f / static_cast<float>(Spark::WIDTH), 1.0f / static_cast<float>(Spark::HEIGHT)};
    motionBlurShaderS->setVec2("texelSize", texelSize);
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

    toneMappingShader.lock()->use();
    toneMappingShader.lock()->setVec2("inversedScreenSize", {1.0f / Spark::WIDTH, 1.0f / Spark::HEIGHT});

    glBindTextureUnit(0, textureHandle);
    glBindTextureUnit(1, averageLuminance);
    screenQuad.draw();
    glBindTextures(0, 2, nullptr);

    textureHandle = toneMappingTexture;
    POP_DEBUG_GROUP();
}

void SparkRenderer::calculateAverageLuminance()
{
    oneOverLogLuminanceRange = 1.0f / logLuminanceRange;

    //this buffer is attached to both shaders in method SparkRenderer::updateBufferBindings()
    luminanceHistogram.update(std::vector<uint32_t>(256)); // resetting histogram buffer

//first compute dispatch
    const auto luminanceHistogramComputeShader = ResourceManager::getInstance()->getShader(ShaderType::LUMINANCE_HISTOGRAM_COMPUTE_SHADER);
    luminanceHistogramComputeShader->use();

    luminanceHistogramComputeShader->setIVec2("inputTextureSize", glm::ivec2(Spark::WIDTH, Spark::HEIGHT));
    luminanceHistogramComputeShader->setFloat("minLogLuminance", minLogLuminance);
    luminanceHistogramComputeShader->setFloat("oneOverLogLuminanceRange", oneOverLogLuminanceRange);

    glBindImageTexture(0, lightColorTexture, 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA16F);
    luminanceHistogramComputeShader->dispatchCompute(Spark::WIDTH / 16, Spark::HEIGHT / 16, 1); //localWorkGroups has dimensions of x = 16, y = 16
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

//second compute dispatch
    const auto averageLuminanceComputeShader = ResourceManager::getInstance()->getShader(ShaderType::AVERAGE_LUMINANCE_COMPUTE_SHADER);
    averageLuminanceComputeShader->use();

    averageLuminanceComputeShader->setUInt("pixelCount", Spark::WIDTH * Spark::HEIGHT);
    averageLuminanceComputeShader->setFloat("minLogLuminance", minLogLuminance);
    averageLuminanceComputeShader->setFloat("logLuminanceRange", logLuminanceRange);
    averageLuminanceComputeShader->setFloat("deltaTime", static_cast<float>(Clock::getDeltaTime()));
    averageLuminanceComputeShader->setFloat("tau", tau);

    glBindImageTexture(0, averageLuminance, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R16F);
    averageLuminanceComputeShader->dispatchCompute(1, 1, 1);  //localWorkGroups has dimensions of x = 16, y = 16
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
}

void SparkRenderer::renderToScreen() const
{
    PUSH_DEBUG_GROUP(RENDER_TO_SCREEN);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_DEPTH_TEST);

    screenShader.lock()->use();

    glBindTextureUnit(0, textureHandle);

    screenQuad.draw();
    POP_DEBUG_GROUP();
}

void SparkRenderer::createFrameBuffersAndTextures()
{
    dofPass->recreateWithNewSize(Spark::WIDTH, Spark::HEIGHT);
    ssaoBlurPass->recreateWithNewSize(Spark::WIDTH / 2, Spark::HEIGHT / 2);

    utils::createTexture(colorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
    utils::createTexture(normalsTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
    utils::createTexture(depthTexture, Spark::WIDTH, Spark::HEIGHT, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

    utils::createTexture(lightColorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(lightDiffuseTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(lightSpecularTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(toneMappingTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(motionBlurTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(lightShaftTexture, Spark::WIDTH / 2, Spark::HEIGHT / 2, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(ssaoTexture, Spark::WIDTH, Spark::HEIGHT, GL_RED, GL_RED, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(fxaaTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::createFramebuffer(mainFramebuffer, {colorTexture, normalsTexture});
    utils::bindDepthTexture(mainFramebuffer, depthTexture);
    utils::createFramebuffer(lightFrameBuffer, {lightColorTexture, lightDiffuseTexture, lightSpecularTexture});
    utils::bindDepthTexture(lightFrameBuffer, depthTexture);
    utils::createFramebuffer(toneMappingFramebuffer, {toneMappingTexture});
    utils::createFramebuffer(motionBlurFramebuffer, {motionBlurTexture});
    utils::createFramebuffer(lightShaftFramebuffer, {lightShaftTexture});
    utils::createFramebuffer(fxaaFramebuffer, {fxaaTexture});

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
}

void SparkRenderer::deleteFrameBuffersAndTextures() const
{
    GLuint textures[9] = {colorTexture,       normalsTexture,    lightColorTexture, lightDiffuseTexture, lightSpecularTexture,
                          toneMappingTexture, motionBlurTexture, depthTexture,      ssaoTexture };
    glDeleteTextures(9, textures);

    GLuint frameBuffers[5] = {mainFramebuffer, lightFrameBuffer, toneMappingFramebuffer, motionBlurFramebuffer, ssaoFramebuffer};

    glDeleteFramebuffers(5, frameBuffers);
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
