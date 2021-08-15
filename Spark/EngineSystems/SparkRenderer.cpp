#include "SparkRenderer.h"

#include "Camera.h"
#include "CommonUtils.h"
#include "DeferredRenderer.hpp"
#include "ForwardPlusRenderer.hpp"
#include "lights/LightProbe.h"
#include "RenderingRequest.h"
#include "ResourceLibrary.h"
#include "Shader.h"
#include "Spark.h"
#include "TileBasedDeferredRenderer.hpp"
#include "TileBasedForwardPlusRenderer.hpp"
#include "ClusterBasedForwardPlusRenderer.hpp"
#include "ClusterBasedDeferredRenderer.hpp"
#include "Timer.h"

namespace spark
{
void SparkRenderer::drawGui()
{
    if(ImGui::BeginMenu("Renderer"))
    {
        const std::string renderingAlgorithmTypeMenu = "Rendering Algorithms";
        if(ImGui::BeginMenu(renderingAlgorithmTypeMenu.c_str()))
        {
            static int mode = 2;
            
            if(ImGui::RadioButton("Deferred", mode == 0))
            {
                mode = 0;
                renderer = std::make_unique<DeferredRenderer>(width, height, cameraUBO, scene.lock()->lightManager);
            }
            if(ImGui::RadioButton("Forward Plus", mode == 1))
            {
                mode = 1;
                renderer = std::make_unique<ForwardPlusRenderer>(width, height, cameraUBO, scene.lock()->lightManager);
            }
            if (ImGui::RadioButton("Tile Based Deferred", mode == 2))
            {
                mode = 2;
                renderer = std::make_unique<TileBasedDeferredRenderer>(width, height, cameraUBO, scene.lock()->lightManager);
            }
            if (ImGui::RadioButton("Tile Based Forward Plus", mode == 3))
            {
                mode = 3;
                renderer = std::make_unique<TileBasedForwardPlusRenderer>(width, height, cameraUBO, scene.lock()->lightManager);
            }
            if (ImGui::RadioButton("Cluster Based Deferred", mode == 4))
            {
                mode = 4;
                renderer = std::make_unique<ClusterBasedDeferredRenderer>(width, height, cameraUBO, scene.lock()->lightManager);
            }
            if (ImGui::RadioButton("Cluster Based Forward Plus", mode == 5))
            {
                mode = 5;
                renderer = std::make_unique<ClusterBasedForwardPlusRenderer>(width, height, cameraUBO, scene.lock()->lightManager);
            }
            ImGui::EndMenu();
        }

        const std::string menuName = "SSAO";
        if(ImGui::BeginMenu(menuName.c_str()))
        {
            ImGui::Checkbox("SSAO enabled", &renderer->isAmbientOcclusionEnabled);
            ImGui::DragInt("Samples", &renderer->ao.kernelSize, 1, 0, 64);
            ImGui::DragFloat("Radius", &renderer->ao.radius, 0.05f, 0.0f);
            ImGui::DragFloat("Bias", &renderer->ao.bias, 0.005f);
            ImGui::DragFloat("Power", &renderer->ao.power, 0.05f, 0.0f);
            ImGui::EndMenu();
        }

        postProcessingStack.drawGui();
        ImGui::EndMenu();
    }
}

SparkRenderer::SparkRenderer(unsigned int windowWidth, unsigned int windowHeight, const std::shared_ptr<Scene>& scene_)
    : scene(scene_), width(windowWidth), height(windowHeight), lightProbesRenderer(scene_->lightManager),
      postProcessingStack(windowWidth, windowHeight, cameraUBO)
{
    renderer = std::make_unique<TileBasedDeferredRenderer>(windowWidth, windowHeight, cameraUBO, scene_->lightManager);

    screenShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("screen.glsl");
    solidColorShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("solidColor.glsl");
    solidColorShader->bindUniformBuffer("Camera", cameraUBO);

    createFrameBuffersAndTextures();
    updateLightBuffersBindings();
}

void SparkRenderer::updateLightBuffersBindings()
{
    const auto& lightManager = scene.lock()->lightManager;
    lightProbesRenderer.bindLightBuffers(lightManager);
    renderer->bindLightBuffers(lightManager);
}

SparkRenderer::~SparkRenderer()
{
    deleteFrameBuffersAndTextures();
}

void SparkRenderer::renderPass()
{
    lightProbesRenderer.process(renderQueue, pbrCubemap.lock(), scene.lock()->lightManager->getLightProbes());

    const auto& camera = scene.lock()->getCamera();
    utils::updateCameraUBO(cameraUBO, camera->getProjectionReversedZ(), camera->getViewMatrix(), camera->getPosition(), camera->getNearPlane(), camera->getFarPlane());

    const GLuint lightingTexture = renderer->process(renderQueue, pbrCubemap, cameraUBO);
    textureHandle = postProcessingStack.process(lightingTexture, renderer->getDepthTexture(), pbrCubemap, scene.lock()->getCamera(), cameraUBO);

    helperShapes();
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

void SparkRenderer::resize(unsigned int windowWidth, unsigned int windowHeight)
{
    width = windowWidth;
    height = windowHeight;
    postProcessingStack.resize(width, height);
    renderer->resize(width, height);
    createFrameBuffersAndTextures();
}

void SparkRenderer::helperShapes()
{
    if(renderQueue[ShaderType::COLOR_ONLY].empty())
        return;

    PUSH_DEBUG_GROUP(HELPER_SHAPES);
    utils::bindTexture2D(uiShapesFramebuffer, textureHandle);
    utils::bindDepthTexture(uiShapesFramebuffer, renderer->getDepthTexture());
    glBindFramebuffer(GL_FRAMEBUFFER, uiShapesFramebuffer);

    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    // light texture must be bound here
    solidColorShader->use();

    for(auto& request : renderQueue[ShaderType::COLOR_ONLY])
    {
        request.mesh->draw(solidColorShader, request.model);
    }

    glEnable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    POP_DEBUG_GROUP();
}

void SparkRenderer::renderToScreen()
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
    utils::recreateFramebuffer(uiShapesFramebuffer);
}

void SparkRenderer::deleteFrameBuffersAndTextures()
{
    GLuint frameBuffers[1] = {uiShapesFramebuffer};

    glDeleteFramebuffers(1, frameBuffers);
}
}  // namespace spark
