#pragma once

#include <memory>

#include "glad_glfw3.h"
#include "ScreenQuad.hpp"
#include "utils/GlHandle.hpp"

namespace spark::resources
{
class Shader;
}

namespace spark::effects
{
class BlurPass
{
    public:
    void blurTexture(GLuint texture) const;
    GLuint getBlurredTexture() const;
    void resize(unsigned int width, unsigned int height);

    BlurPass(unsigned int width_, unsigned int height_);
    ~BlurPass();

    BlurPass& operator=(const BlurPass& blurPass) = delete;
    BlurPass& operator=(const BlurPass&& blurPass) = delete;
    BlurPass(const BlurPass& blurPass) = delete;
    BlurPass(const BlurPass&& blurPass) = delete;

    private:
    unsigned int w{}, h{};
    utils::TextureHandle hTexture{}, vTexture{};
    GLuint vFramebuffer{};
    ScreenQuad screenQuad{};

    std::shared_ptr<resources::Shader> horizontalGaussianBlurShader{nullptr};
    std::shared_ptr<resources::Shader> verticalGaussianBlurShader{nullptr};

    void createGlObjects();
};
}  // namespace spark::effects