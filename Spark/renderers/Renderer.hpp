#pragma once

#include "LightProbesRenderer.hpp"
#include "effects/AmbientOcclusion.hpp"
#include "Scene.h"
#include "effects/PostProcessingStack.hpp"

namespace spark::renderers
{
class Renderer
{
    public:
    Renderer(unsigned int width, unsigned int height);
    Renderer(const Renderer&) = delete;
    Renderer(Renderer&&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    Renderer& operator=(Renderer&&) = delete;
    virtual ~Renderer();

    void render(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera, unsigned int width, unsigned int height,
                unsigned int outputFramebuffer = 0);
    void render(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera, unsigned int x, unsigned int y, unsigned int width,
                unsigned int height, unsigned int outputFramebuffer = 0);
    void resize(unsigned int width, unsigned int height);

    void drawGui();
    unsigned int getWidth() const;
    unsigned int getHeight() const;

    bool isAmbientOcclusionEnabled{false};
    bool isProfilingEnabled{false};

    protected:
    virtual void renderMeshes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera) = 0;
    void postProcessingPass(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera);

    virtual void resizeDerived(unsigned int width, unsigned int height) = 0;
    void resizeBase(unsigned int width, unsigned int height);

    virtual GLuint getDepthTexture() const = 0;
    virtual GLuint getLightingTexture() const = 0;

    unsigned int w{}, h{};
    effects::AmbientOcclusion ao;

    private:
    void helperShapes(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera);
    void renderToFramebuffer(unsigned int outputFramebuffer, unsigned int x, unsigned int y, unsigned int width, unsigned int height) const;
    void clearFramebuffer(unsigned int outputFramebuffer, unsigned int width, unsigned int height) const;

    effects::PostProcessingStack postProcessingStack;
    renderers::LightProbesRenderer lightProbesRenderer;

    ScreenQuad screenQuad{};
    GLuint uiShapesFramebuffer{};

    GLuint textureHandle{};  // temporary, its only a handle to other texture -> dont delete it

    std::shared_ptr<resources::Shader> screenShader{nullptr};
    std::shared_ptr<resources::Shader> solidColorShader{nullptr};
};

}  // namespace spark::renderers