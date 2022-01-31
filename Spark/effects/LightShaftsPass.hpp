#pragma once

#include <memory>
#include <optional>

#include "BlurPass.h"
#include "glad_glfw3.h"
#include "ScreenQuad.hpp"
#include "lights/DirectionalLight.h"
#include "utils/GlHandle.hpp"

namespace spark
{
class ICamera;
}

namespace spark::resources
{
class Shader;
}

namespace spark::effects
{
class LightShaftsPass
{
    public:
    LightShaftsPass(unsigned int width, unsigned int height);
    LightShaftsPass(const LightShaftsPass&) = delete;
    LightShaftsPass(LightShaftsPass&&) = delete;
    LightShaftsPass& operator=(const LightShaftsPass&) = delete;
    LightShaftsPass& operator=(LightShaftsPass&&) = delete;
    ~LightShaftsPass();

    std::optional<GLuint> process(const std::shared_ptr<ICamera>& camera, GLuint depthTexture, GLuint lightingTexture);
    void resize(unsigned int width, unsigned int height);

    float exposure = 0.004f;
    float decay = 0.970f;
    float density = 0.45f;
    float weight = 8.0f;

    private:
    void createFrameBuffersAndTextures();
    void renderLightShaftsToTexture(const lights::DirectionalLight* const dirLight, GLuint depthTexture, GLuint lightingTexture,
                                    const glm::vec2 lightScreenPos) const;
    void blurLightShafts() const;
    void blendLightShafts(GLuint lightingTexture) const;
    static glm::vec2 dirLightPositionInScreenSpace(const std::shared_ptr<ICamera>& camera, const lights::DirectionalLight* const dirLight);
    static bool isCameraFacingDirectionalLight(glm::vec2 dirLightScreenSpacePosition, const std::shared_ptr<ICamera>& camera,
                                               const lights::DirectionalLight* const dirLight);

    unsigned int w{}, h{};
    GLuint radialBlurFramebuffer1{};
    utils::TextureHandle radialBlurTexture1{}, radialBlurTexture2{}, blendingOutputTexture{};
    GLuint blendingFramebuffer{};
    ScreenQuad screenQuad{};
    BlurPass blurPass;
    std::shared_ptr<resources::Shader> lightShaftsShader{nullptr};
    std::shared_ptr<resources::Shader> blendingShader{nullptr};
};
}  // namespace spark::effects