#pragma once

#include <memory>
#include <optional>

#include "Buffer.hpp"
#include "Cube.hpp"
#include "glad_glfw3.h"
#include "Scene.h"
#include "TexturePass.hpp"

namespace spark
{
struct PbrCubemapTexture;
}

namespace spark::resources
{
class Shader;
}

namespace spark::effects
{
class SkyboxPass
{
    public:
    SkyboxPass(unsigned int width, unsigned int height);
    SkyboxPass(const SkyboxPass&) = delete;
    SkyboxPass(SkyboxPass&&) = delete;
    SkyboxPass& operator=(const SkyboxPass&) = delete;
    SkyboxPass& operator=(SkyboxPass&&) = delete;
    ~SkyboxPass();

    std::optional<GLuint> process(GLuint depthTexture, GLuint lightingTexture, const std::shared_ptr<Scene>& scene);
    void processFramebuffer(const std::weak_ptr<PbrCubemapTexture>& cubemap, GLuint framebuffer, unsigned int fboWidth, unsigned int fboHeight,
                            const UniformBuffer& cameraUbo);
    void renderSkybox(GLuint framebuffer, unsigned int fboWidth, unsigned int fboHeight, const std::shared_ptr<PbrCubemapTexture>& cubemapPtr,
                      const UniformBuffer& cameraUbo);
    void resize(unsigned int width, unsigned int height);

    private:
    void createFrameBuffersAndTextures();

    unsigned int w{}, h{};

    Cube cube{};
    GLuint cubemapFramebuffer{}, cubemapTexture{};
    TexturePass texturePass;
    std::shared_ptr<resources::Shader> cubemapShader{nullptr};
};
}  // namespace spark::effects
