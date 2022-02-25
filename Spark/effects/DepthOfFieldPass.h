#pragma once

#include <memory>

#include "Buffer.hpp"
#include "glad_glfw3.h"
#include "ScreenQuad.hpp"
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
class BlurPass;

class DepthOfFieldPass
{
    public:
    GLuint process(GLuint lightPassTexture, GLuint depthTexture, const std::shared_ptr<ICamera>& camera) const;
    void resize(unsigned int width, unsigned int height);

    DepthOfFieldPass(unsigned int width, unsigned int height);
    DepthOfFieldPass& operator=(const DepthOfFieldPass& blurPass) = delete;
    DepthOfFieldPass& operator=(const DepthOfFieldPass&& blurPass) = delete;
    DepthOfFieldPass(const DepthOfFieldPass& blurPass) = delete;
    DepthOfFieldPass(const DepthOfFieldPass&& blurPass) = delete;
    ~DepthOfFieldPass();

    float aperture{0.8f}, f{0.035f};
    float focusPoint{20.0f}, maxCoC{0.035};
    float poissonBlurScale{0.1};

    private:
    void createFrameBuffersAndTextures();

    void calculateCircleOfConfusion(GLuint lightingTexture, GLuint depthTexture, const std::shared_ptr<ICamera>& camera) const;
    void blurLightPassTexture(GLuint depthTexture) const;
    void blendDepthOfField(GLuint lightPassTexture) const;

    unsigned int w{}, h{};

    std::shared_ptr<resources::Shader> cocShader{nullptr};
    std::shared_ptr<resources::Shader> poissonBlurShader{nullptr};
    std::shared_ptr<resources::Shader> poissonBlurShader2{nullptr};
    std::shared_ptr<resources::Shader> blendShader{nullptr};

    utils::UniqueTextureHandle poissonBlurTexture{}, poissonBlurTexture2;
    utils::TextureHandle cocTexture{}, blendDofTexture{};
    GLuint cocFramebuffer{}, blendDofFramebuffer{}, poissonBlurFramebuffer{}, poissonBlurFramebuffer2{};

    ScreenQuad screenQuad{};
    SSBO taps16{};
    SSBO taps16_2{};
};
}  // namespace spark::effects