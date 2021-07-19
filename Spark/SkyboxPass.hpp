#pragma once

#include <memory>
#include <optional>

#include "Buffer.hpp"
#include "glad_glfw3.h"
#include "ScreenQuad.hpp"
#include "Structs.h"

namespace spark
{
namespace resources
{
    class Shader;
}

class SkyboxPass
{
    public:
    SkyboxPass() = default;
    SkyboxPass(const SkyboxPass&) = delete;
    SkyboxPass(SkyboxPass&&) = delete;
    SkyboxPass& operator=(const SkyboxPass&) = delete;
    SkyboxPass& operator=(SkyboxPass&&) = delete;
    ~SkyboxPass();

    void setup(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo);
    std::optional<GLuint> process(const std::weak_ptr<PbrCubemapTexture>& cubemap, GLuint depthTexture, GLuint lightingTexture);
    void processFramebuffer(const std::weak_ptr<PbrCubemapTexture>& cubemap, GLuint framebuffer, unsigned int fboWidth, unsigned int fboHeight);
    void renderSkybox(GLuint framebuffer, unsigned fboWidth, unsigned fboHeight, const std::shared_ptr<PbrCubemapTexture>& cubemapPtr);
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);
    void cleanup();

    private:
    unsigned int w{}, h{};

    Cube cube = Cube();
    GLuint cubemapFramebuffer{};
    std::shared_ptr<resources::Shader> cubemapShader{nullptr};
};
}  // namespace spark
