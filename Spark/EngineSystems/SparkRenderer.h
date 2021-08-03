#pragma once

#include <glad_glfw3.h>

#include "Enums.h"
#include "effects/PostProcessingStack.hpp"
#include "LightProbesRenderer.hpp"
#include "Scene.h"
#include "ScreenQuad.hpp"
#include "RenderingRequest.h"

namespace spark
{
class Renderer;

namespace lights
{
    class LightProbe;
}
struct PbrCubemapTexture;

class SparkRenderer
{
    public:
    SparkRenderer(unsigned int windowWidth, unsigned int windowHeight, const std::shared_ptr<Scene>& scene_);
    SparkRenderer() = delete;
    SparkRenderer(const SparkRenderer&) = delete;
    SparkRenderer(SparkRenderer&&) = delete;
    SparkRenderer& operator=(const SparkRenderer&) = delete;
    SparkRenderer& operator=(SparkRenderer&&) = delete;
    ~SparkRenderer();

    void renderPass();

    void drawGui();
    void addRenderingRequest(const RenderingRequest& request);
    void setScene(const std::shared_ptr<Scene>& scene_);
    void setCubemap(const std::shared_ptr<PbrCubemapTexture>& cubemap);
    void resize(unsigned int windowWidth, unsigned int windowHeight);

    private:
    void updateLightBuffersBindings();

    void helperShapes();
    void renderToScreen();
    void clearRenderQueues();
    void createFrameBuffersAndTextures();
    void deleteFrameBuffersAndTextures();

    UniformBuffer cameraUBO{};
    std::map<ShaderType, std::deque<RenderingRequest>> renderQueue{};
    std::weak_ptr<Scene> scene{};

    unsigned int width{}, height{};
    std::weak_ptr<PbrCubemapTexture> pbrCubemap;
    std::unique_ptr<Renderer> renderer;
    LightProbesRenderer lightProbesRenderer;
    effects::PostProcessingStack postProcessingStack;

    ScreenQuad screenQuad{};
    GLuint uiShapesFramebuffer{};

    GLuint textureHandle{};  // temporary, its only a handle to other texture -> dont delete it

    std::shared_ptr<resources::Shader> screenShader{nullptr};
    std::shared_ptr<resources::Shader> solidColorShader{nullptr};
};
}  // namespace spark