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

#include "Camera.h"
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
#define PUSH_DEBUG_GROUP(x, y)                                                                                    \
    {                                                                                                             \
        std::string message = #x;                                                                                 \
        glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, y, static_cast<GLsizei>(message.length()), message.data()); \
    }

#define POP_DEBUG_GROUP() glPopDebugGroup()

SparkRenderer* SparkRenderer::getInstance()
{
    static SparkRenderer* spark_renderer = nullptr;
    if(spark_renderer == nullptr)
    {
        spark_renderer = new SparkRenderer();
    }
    return spark_renderer;
}

void SparkRenderer::updateBufferBindings() const
{
    const std::shared_ptr<Shader> lShader = lightShader.lock();
    lShader->use();
    lShader->bindSSBO("DirLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->dirLightSSBO);
    lShader->bindSSBO("PointLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->pointLightSSBO);
    lShader->bindSSBO("SpotLightData", SceneManager::getInstance()->getCurrentScene()->lightManager->spotLightSSBO);

    mainShader.lock()->bindUniformBuffer("Camera", uniformBuffer);
    lightShader.lock()->bindUniformBuffer("Camera", uniformBuffer);
    motionBlurShader.lock()->bindUniformBuffer("Camera", uniformBuffer);
    cubemapShader.lock()->bindUniformBuffer("Camera", uniformBuffer);
    ssaoShader.lock()->bindUniformBuffer("Camera", uniformBuffer);
    ssaoShader.lock()->bindUniformBuffer("Samples", sampleUniformBuffer);
}

void SparkRenderer::drawGui()
{
    if(ImGui::BeginMenu("SparkRenderer"))
    {
        std::string menuName = "SSAO";
        if(ImGui::BeginMenu(menuName.c_str()))
        {
            ImGui::Checkbox("SSAO enabled", &ssaoEnable);
            ImGui::DragInt("Samples", &kernelSize, 1, 0, 64);
            ImGui::DragFloat("Radius", &radius, 0.05f, 0.0f);
            ImGui::DragFloat("Bias", &bias, 0.005f);
            ImGui::DragFloat("Power", &power, 0.05f, 0.0f);
            ImGui::EndMenu();
        }
        ImGui::EndMenu();
    }
}

void SparkRenderer::setup()
{
    uniformBuffer.genBuffer();
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
    postprocessingShader = ResourceManager::getInstance()->getShader(ShaderType::POSTPROCESSING_SHADER);
    lightShader = ResourceManager::getInstance()->getShader(ShaderType::LIGHT_SHADER);
    motionBlurShader = ResourceManager::getInstance()->getShader(ShaderType::MOTION_BLUR_SHADER);
    cubemapShader = ResourceManager::getInstance()->getShader(ShaderType::CUBEMAP_SHADER);
    ssaoShader = ResourceManager::getInstance()->getShader(ShaderType::SSAO_SHADER);

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
            uniformBuffer.update<CamData>({camData});
            camera->cleanDirty();
        }
    }
    fillGBuffer();
    ssaoComputing();
    renderLights();
    renderCubemap();
    // lightShafts();
    postprocessingPass();
    motionBlur();
    renderToScreen();

    PUSH_DEBUG_GROUP(GUI, 0);
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
    PUSH_DEBUG_GROUP(RENDER_TO_MAIN_FRAMEBUFFER, 0);
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
    PUSH_DEBUG_GROUP(SSAO, 0);

    if(!ssaoEnable)
    {
        glBindFramebuffer(GL_FRAMEBUFFER, gaussianBlurFramebuffer2);
        glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        POP_DEBUG_GROUP();
        return;
    }

    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER, ssaoFramebuffer);
    glClearColor(0, 0, 0, 0);
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

    blurTexture(ssaoTexture);
    glEnable(GL_DEPTH_TEST);

    POP_DEBUG_GROUP();
}

void SparkRenderer::renderLights() const
{
    PUSH_DEBUG_GROUP(PBR_LIGHT, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, lightFrameBuffer);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    const auto cubemap = SceneManager::getInstance()->getCurrentScene()->cubemap;

    glDisable(GL_DEPTH_TEST);
    const std::shared_ptr<Shader> lShader = lightShader.lock();
    lShader->use();
    if(cubemap)
    {
        std::array<GLuint, 7> textures{
            depthTexture,       colorTexture, normalsTexture, cubemap->irradianceCubemap, cubemap->prefilteredCubemap, cubemap->brdfLUTTexture,
            verticalBlurTexture};
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
    }
    else
    {
        std::array<GLuint, 7> textures{depthTexture, colorTexture, normalsTexture, 0, 0, 0, verticalBlurTexture};
        glBindTextures(0, static_cast<GLsizei>(textures.size()), textures.data());
        screenQuad.draw();
        glBindTextures(0, static_cast<GLsizei>(textures.size()), nullptr);
    }
    glEnable(GL_DEPTH_TEST);

    POP_DEBUG_GROUP();
}

void SparkRenderer::renderCubemap() const
{
    const auto cubemap = SceneManager::getInstance()->getCurrentScene()->cubemap;
    if(!cubemap)
        return;

    PUSH_DEBUG_GROUP(RENDER_CUBEMAP, 0);

    glDepthFunc(GL_GEQUAL);
    const auto cubemapShaderPtr = cubemapShader.lock();
    cubemapShaderPtr->use();

    glBindTextureUnit(0, cubemap->cubemap);
    cube.draw();
    glBindTextures(0, 1, nullptr);
    glDepthFunc(GL_GREATER);

    POP_DEBUG_GROUP();
}

void SparkRenderer::blurTexture(GLuint texture) const
{
    PUSH_DEBUG_GROUP(GAUSSIAN_BLUR, 0);
    glViewport(0, 0, Spark::WIDTH / 2, Spark::HEIGHT / 2);
    glBindFramebuffer(GL_FRAMEBUFFER, gaussianBlurFramebuffer);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    const auto gaussianBlurShader = ResourceManager::getInstance()->getShader(ShaderType::GAUSSIAN_BLUR_SHADER);
    gaussianBlurShader->use();
    gaussianBlurShader->setVec2("inverseScreenSize", {1.0f / ((float)Spark::WIDTH / 2), 1.0f / ((float)Spark::HEIGHT / 2)});
    gaussianBlurShader->setVec2("direction", {1.0f, 0.0f});
    glBindTextureUnit(0, texture);
    screenQuad.draw();

    glBindFramebuffer(GL_FRAMEBUFFER, gaussianBlurFramebuffer2);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    gaussianBlurShader->setVec2("direction", {0.0f, 1.0f});
    glBindTextureUnit(0, horizontalBlurTexture);
    screenQuad.draw();
    glBindTextures(0, 1, nullptr);
    glViewport(0, 0, Spark::WIDTH, Spark::HEIGHT);
    POP_DEBUG_GROUP();
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

    glBindFramebuffer(GL_FRAMEBUFFER, gaussianBlurFramebuffer2);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    if(lightScreenPos.x < 0.0f || lightScreenPos.x > 1.0f || lightScreenPos.y < 0.0f || lightScreenPos.y > 1.0f)
    {
        return;
    }
    // std::cout << "Light world pos: " << dirLightPosition.x << ", " << dirLightPosition.y << ", " << dirLightPosition.z << std::endl;
    // std::cout << "Light on the screen. Pos: " << lightScreenPos.x << ", "<< lightScreenPos.y<< std::endl;
    PUSH_DEBUG_GROUP(LIGHT SHAFTS, 0);
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
    blurTexture(lightShaftTexture);
    POP_DEBUG_GROUP();
}

void SparkRenderer::postprocessingPass()
{
    PUSH_DEBUG_GROUP(POSTPROCESSING, 0);
    textureHandle = postProcessingTexture;

    glBindFramebuffer(GL_FRAMEBUFFER, postprocessingFramebuffer);
    glDisable(GL_DEPTH_TEST);

    postprocessingShader.lock()->use();
    postprocessingShader.lock()->setVec2("inversedScreenSize", {1.0f / Spark::WIDTH, 1.0f / Spark::HEIGHT});

    glBindTextureUnit(0, lightColorTexture);
    glBindTextureUnit(1, verticalBlurTexture);
    screenQuad.draw();
    glBindTextures(0, 2, nullptr);

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

    textureHandle = motionBlurTexture;

    PUSH_DEBUG_GROUP(MOTION_BLUR, 0);
    const std::shared_ptr<Shader> motionBlurShaderS = motionBlurShader.lock();
    glBindFramebuffer(GL_FRAMEBUFFER, motionBlurFramebuffer);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    motionBlurShaderS->use();
    motionBlurShaderS->setMat4("prevViewProj", prevProjectionView);
    motionBlurShaderS->setFloat("currentFPS", static_cast<float>(Clock::getFPS()));
    const glm::vec2 texelSize = {1.0f / static_cast<float>(Spark::WIDTH), 1.0f / static_cast<float>(Spark::HEIGHT)};
    motionBlurShaderS->setVec2("texelSize", texelSize);
    std::array<GLuint, 2> textures2{postProcessingTexture, depthTexture};
    glBindTextures(0, static_cast<GLsizei>(textures2.size()), textures2.data());
    screenQuad.draw();
    glBindTextures(0, static_cast<GLsizei>(textures2.size()), nullptr);

    prevProjectionView = projectionView;
    POP_DEBUG_GROUP();
}

void SparkRenderer::renderToScreen() const
{
    PUSH_DEBUG_GROUP(RENDER_TO_SCREEN, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_DEPTH_TEST);

    screenShader.lock()->use();

    glBindTextureUnit(0, textureHandle);

    screenQuad.draw();
    POP_DEBUG_GROUP();
}

void SparkRenderer::createFrameBuffersAndTextures()
{
    createTexture(colorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
    createTexture(normalsTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);
    createTexture(depthTexture, Spark::WIDTH, Spark::HEIGHT, GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_NEAREST);

    createTexture(lightColorTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    createTexture(postProcessingTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    createTexture(motionBlurTexture, Spark::WIDTH, Spark::HEIGHT, GL_RGB, GL_RGB, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    createTexture(lightShaftTexture, Spark::WIDTH / 2, Spark::HEIGHT / 2, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    createTexture(horizontalBlurTexture, Spark::WIDTH / 2, Spark::HEIGHT / 2, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    createTexture(verticalBlurTexture, Spark::WIDTH / 2, Spark::HEIGHT / 2, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    createTexture(ssaoTexture, Spark::WIDTH, Spark::HEIGHT, GL_RED, GL_RED, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);

    createFramebuffer(mainFramebuffer, {colorTexture, normalsTexture});
    bindDepthTexture(mainFramebuffer, depthTexture);
    createFramebuffer(lightFrameBuffer, {lightColorTexture});
    bindDepthTexture(lightFrameBuffer, depthTexture);
    createFramebuffer(postprocessingFramebuffer, {postProcessingTexture});
    createFramebuffer(motionBlurFramebuffer, {motionBlurTexture});
    createFramebuffer(lightShaftFramebuffer, {lightShaftTexture});
    createFramebuffer(gaussianBlurFramebuffer, {horizontalBlurTexture});
    createFramebuffer(gaussianBlurFramebuffer2, {verticalBlurTexture});
    createFramebuffer(ssaoFramebuffer, {ssaoTexture});
}

void SparkRenderer::createFramebuffer(GLuint& framebuffer, const std::vector<GLuint>&& colorTextures, GLuint renderbuffer)
{
    glCreateFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    std::vector<GLenum> colorAttachments;
    colorAttachments.reserve(colorTextures.size());
    for(unsigned int i = 0; i < colorTextures.size(); ++i)
    {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorTextures[i], 0);
        colorAttachments.push_back(GL_COLOR_ATTACHMENT0 + i);
    }
    glDrawBuffers(static_cast<GLsizei>(colorAttachments.size()), colorAttachments.data());

    if(renderbuffer != 0)
    {
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderbuffer);
    }

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        throw std::exception("Framebuffer incomplete!");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SparkRenderer::bindDepthTexture(GLuint& framebuffer, GLuint depthTexture)
{
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void SparkRenderer::createTexture(GLuint& texture, GLuint width, GLuint height, GLenum internalFormat, GLenum format, GLenum pixelFormat,
                                  GLenum textureWrapping, GLenum textureSampling)
{
    glCreateTextures(GL_TEXTURE_2D, 1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, pixelFormat, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, textureSampling);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, textureSampling);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, textureWrapping);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, textureWrapping);
}

void SparkRenderer::cleanup()
{
    uniformBuffer.cleanup();
    sampleUniformBuffer.cleanup();
    deleteFrameBuffersAndTextures();
    glDeleteTextures(1, &randomNormalsTexture);
}

void SparkRenderer::deleteFrameBuffersAndTextures() const
{
    GLuint textures[9] = {colorTexture,          normalsTexture,      lightColorTexture, postProcessingTexture, motionBlurTexture, depthTexture,
                          horizontalBlurTexture, verticalBlurTexture, ssaoTexture};
    glDeleteTextures(9, textures);

    GLuint frameBuffers[7] = {mainFramebuffer,          lightFrameBuffer, postprocessingFramebuffer, motionBlurFramebuffer, gaussianBlurFramebuffer,
                              gaussianBlurFramebuffer2, ssaoFramebuffer};

    glDeleteFramebuffers(7, frameBuffers);
}

}  // namespace spark
