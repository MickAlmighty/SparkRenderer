#pragma once

#include <memory>

#include "glad_glfw3.h"
#include "ScreenQuad.hpp"

namespace spark::resources
{
class Shader;
}

namespace spark::effects
{
class FxaaPass
{
    public:
    FxaaPass() = default;
    FxaaPass(const FxaaPass&) = delete;
    FxaaPass(FxaaPass&&) = delete;
    FxaaPass& operator=(const FxaaPass&) = delete;
    FxaaPass& operator=(FxaaPass&&) = delete;
    ~FxaaPass();

    void setup(unsigned int width, unsigned int height);
    GLuint process(GLuint inputTexture);
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);
    void cleanup();

    private:
    ScreenQuad screenQuad{};
    GLuint fxaaFramebuffer{}, fxaaTexture{};
    std::shared_ptr<resources::Shader> fxaaShader{nullptr};
};
}  // namespace spark::effects
