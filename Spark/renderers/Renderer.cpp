#include "Renderer.hpp"

#include "CommonUtils.h"
#include "effects/PostProcessingStack.hpp"
#include "LightProbesRenderer.hpp"
#include "RenderingRequest.h"
#include "Spark.h"

namespace spark::renderers
{
Renderer::Renderer(unsigned int width, unsigned int height) : w(width), h(height), ao(w, h), postProcessingStack(w, h)
{
    screenShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("screen.glsl");
    solidColorShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("solidColor.glsl");

    utils::createFramebuffer(uiShapesFramebuffer);
}

Renderer::~Renderer()
{
    glDeleteFramebuffers(1, &uiShapesFramebuffer);
}

void Renderer::render(const std::shared_ptr<Scene>& scene)
{
    lightProbesRenderer.process(scene);
    renderMeshes(scene);
    postProcessingPass(scene);

    helperShapes(scene);
    renderToScreen();
    // render to screen
}

void Renderer::resize(unsigned int width, unsigned int height)
{
    resizeBase(width, height);
    resizeDerived(width, height);
}

void Renderer::drawGui()
{
    const std::string menuName = "SSAO";
    if(ImGui::BeginMenu(menuName.c_str()))
    {
        ImGui::Checkbox("SSAO enabled", &isAmbientOcclusionEnabled);
        ImGui::DragInt("Samples", &ao.kernelSize, 1, 0, 64);
        ImGui::DragFloat("Radius", &ao.radius, 0.05f, 0.0f);
        ImGui::DragFloat("Bias", &ao.bias, 0.005f);
        ImGui::DragFloat("Power", &ao.power, 0.05f, 0.0f);
        ImGui::EndMenu();
    }

    postProcessingStack.drawGui();
}

void Renderer::postProcessingPass(const std::shared_ptr<Scene>& scene)
{
    textureHandle = postProcessingStack.process(getLightingTexture(), getDepthTexture(), scene);
}

void Renderer::resizeBase(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    ao.resize(w, h);
    postProcessingStack.resize(w, h);
}

void Renderer::helperShapes(const std::shared_ptr<Scene>& scene)
{
    const auto it = scene->getRenderingQueues().find(ShaderType::COLOR_ONLY);
    if(it == scene->getRenderingQueues().end())
        return;

    const auto& [shaderType, renderingQueue] = *it;

    if(renderingQueue.empty())
        return;

    PUSH_DEBUG_GROUP(HELPER_SHAPES)
    utils::bindTexture2D(uiShapesFramebuffer, textureHandle);
    utils::bindDepthTexture(uiShapesFramebuffer, getDepthTexture());
    glBindFramebuffer(GL_FRAMEBUFFER, uiShapesFramebuffer);

    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    // light texture must be bound here
    solidColorShader->use();
    solidColorShader->bindUniformBuffer("Camera", scene->getCamera()->getUbo());

    for(auto& request : renderingQueue)
    {
        request.mesh->draw(solidColorShader, request.model);
    }

    glEnable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    POP_DEBUG_GROUP()
}

void Renderer::renderToScreen()
{
    PUSH_DEBUG_GROUP(RENDER_TO_SCREEN)
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glDisable(GL_DEPTH_TEST);

    screenShader->use();

    glBindTextureUnit(0, textureHandle);

    screenQuad.draw();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    POP_DEBUG_GROUP()
}
}  // namespace spark::renderers