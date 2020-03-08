#include "EngineSystems/SparkRenderer.h"

#include <iostream>
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
#include "HID.h"
#include "ResourceLoader.h"
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
        for(unsigned int i = 0; i < 16; i++)
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

    dofPass = std::make_unique<DepthOfFieldPass>(Spark::WIDTH, Spark::HEIGHT);
    ssaoBlurPass = std::make_unique<BlurPass>(Spark::WIDTH / 2, Spark::HEIGHT / 2);

    updateBufferBindings();
    createFrameBuffersAndTextures();
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
    depthOfField();
    // lightShafts();
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

void SparkRenderer::ssaoComputing() const
{
    PUSH_DEBUG_GROUP(SSAO);

    glBindFramebuffer(GL_FRAMEBUFFER, ssaoBlurPass->getSecondPassFramebuffer());
    glClearColor(1.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if(!ssaoEnable)
    {
        POP_DEBUG_GROUP();
        return;
    }

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
    if(cubemap)
    {
        std::array<GLuint, 7> textures{depthTexture,
                                       colorTexture,
                                       normalsTexture,
                                       cubemap->irradianceCubemap,
                                       cubemap->prefilteredCubemap,
                                       cubemap->brdfLUTTexture,
                                       ssaoBlurPass->getBlurredTexture()};
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
    }
    else
    {
        std::array<GLuint, 7> textures{depthTexture, colorTexture, normalsTexture, 0, 0, 0, ssaoBlurPass->getBlurredTexture()};
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
    if(dirLights.empty())
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

    const auto shader = ResourceManager::getInstance()->getShader(ShaderType::LIGHT_SHAFTS_SHADER);
    shader->use();
    shader->setVec2("lightScreenPos", lightScreenPos);
    shader->setVec3("lightColor", dirLights[0].lock()->getColor());

    glBindTextureUnit(0, depthTexture);
    screenQuad.draw();
    glBindTextureUnit(0, 0);

    glViewport(0, 0, Spark::WIDTH, Spark::HEIGHT);
    // glBindFramebuffer(GL_FRAMEBUFFER, 0);
    ssaoBlurPass->blurTexture(lightShaftTexture);
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

    for(unsigned int counter = 0, i = 0; i < 4; ++i)
    {
        glm::vec4 currentColumn = projectionView[i];
        glm::vec4 previousColumn = prevProjectionView[i];

        if(currentColumn == previousColumn)
        {
            ++counter;
        }
        if(counter == 4)
        {
            return;
        }
    }

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

    glBindFramebuffer(GL_FRAMEBUFFER, toneMappingFramebuffer);
    glDisable(GL_DEPTH_TEST);

    toneMappingShader.lock()->use();
    toneMappingShader.lock()->setVec2("inversedScreenSize", {1.0f / Spark::WIDTH, 1.0f / Spark::HEIGHT});

    glBindTextureUnit(0, textureHandle);
    screenQuad.draw();
    glBindTextures(0, 2, nullptr);

    textureHandle = toneMappingTexture;
    POP_DEBUG_GROUP();
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
    utils::createTexture(depthTexture, Spark::WIDTH, Spark::HEIGHT, GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

    utils::createTexture(lightColorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(toneMappingTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(motionBlurTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(lightShaftTexture, Spark::WIDTH / 2, Spark::HEIGHT / 2, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(ssaoTexture, Spark::WIDTH, Spark::HEIGHT, GL_RED, GL_RED, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::createTexture(fxaaTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::createFramebuffer(mainFramebuffer, {colorTexture, normalsTexture});
    utils::bindDepthTexture(mainFramebuffer, depthTexture);
    utils::createFramebuffer(lightFrameBuffer, {lightColorTexture});
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
}

void SparkRenderer::deleteFrameBuffersAndTextures() const
{
    GLuint textures[7] = {colorTexture, normalsTexture, lightColorTexture, toneMappingTexture, motionBlurTexture, depthTexture, ssaoTexture};
    glDeleteTextures(7, textures);

    GLuint frameBuffers[5] = {mainFramebuffer, lightFrameBuffer, toneMappingFramebuffer, motionBlurFramebuffer, ssaoFramebuffer};

    glDeleteFramebuffers(5, frameBuffers);
}
}  // namespace spark
