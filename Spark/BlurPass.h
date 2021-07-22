#pragma once

#include <memory>

#include "glad_glfw3.h"
#include "ScreenQuad.hpp"

namespace spark
{
namespace resources
{
    class Shader;
}

class BlurPass
{
    public:
    void blurTexture(GLuint texture) const;
    GLuint getBlurredTexture() const;
    void recreateWithNewSize(unsigned int width, unsigned int height);

    BlurPass(unsigned int width_, unsigned int height_);
    ~BlurPass();

    BlurPass& operator=(const BlurPass& blurPass) = delete;
    BlurPass& operator=(const BlurPass&& blurPass) = delete;
    BlurPass(const BlurPass& blurPass) = delete;
    BlurPass(const BlurPass&& blurPass) = delete;

    private:
    unsigned int width{}, height{};
    GLuint hTexture{};
    GLuint vFramebuffer{}, vTexture{};
    ScreenQuad screenQuad{};

    std::shared_ptr<resources::Shader> gaussianBlurShader{nullptr};

    void createGlObjects();
    void deleteGlObjects();
};
}  // namespace spark