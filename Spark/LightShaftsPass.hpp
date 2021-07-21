#pragma once

#include <memory>
#include <optional>

#include "BlurPass.h"
#include "Camera.h"
#include "glad_glfw3.h"
#include "ScreenQuad.hpp"
#include "Lights/DirectionalLight.h"

namespace spark
{
namespace resources
{
    class Shader;
}

class LightShaftsPass
{
    public:
    LightShaftsPass() = default;
    LightShaftsPass(const LightShaftsPass&) = delete;
    LightShaftsPass(LightShaftsPass&&) = delete;
    LightShaftsPass& operator=(const LightShaftsPass&) = delete;
    LightShaftsPass& operator=(LightShaftsPass&&) = delete;
    ~LightShaftsPass();

    void setup(unsigned int width, unsigned int height);
    std::optional<GLuint> process(const std::shared_ptr<Camera>& camera, GLuint depthTexture,
                                  GLuint lightingTexture);
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);
    void cleanup();

    float exposure = 0.004f;
    float decay = 0.960f;
    float density = 1.0f;
    float weight = 6.65f;

    private:
    void renderLightShaftsToTexture(const DirectionalLight* const dirLight, GLuint depthTexture, GLuint lightingTexture, const glm::vec2 lightScreenPos) const;
    void blurLightShafts() const;
    void blendLightShafts(GLuint lightingTexture) const;
    static glm::vec2 dirLightPositionInScreenSpace(const std::shared_ptr<Camera>& camera, const DirectionalLight* const dirLight);
    static bool isCameraFacingDirectionalLight(glm::vec2 dirLightScreenSpacePosition, const std::shared_ptr<Camera>& camera, const DirectionalLight* const dirLight);

    unsigned int w{}, h{};
    GLuint radialBlurFramebuffer1{}, radialBlurTexture1{};
    GLuint radialBlurTexture2{};
    GLuint blendingFramebuffer{}, blendingOutputTexture{};
    ScreenQuad screenQuad{};
    std::unique_ptr<BlurPass> blurPass;
    std::shared_ptr<resources::Shader> lightShaftsShader{nullptr};
    std::shared_ptr<resources::Shader> blendingShader{nullptr};
};
}  // namespace spark