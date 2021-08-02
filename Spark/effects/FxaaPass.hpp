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
    FxaaPass(unsigned int width, unsigned int height);
    FxaaPass(const FxaaPass&) = delete;
    FxaaPass(FxaaPass&&) = delete;
    FxaaPass& operator=(const FxaaPass&) = delete;
    FxaaPass& operator=(FxaaPass&&) = delete;
    ~FxaaPass();

    GLuint process(GLuint inputTexture);
    void resize(unsigned int width, unsigned int height);

    private:
    void createFrameBuffersAndTextures();

    unsigned int w{}, h{};
    ScreenQuad screenQuad{};
    GLuint fxaaFramebuffer{}, fxaaTexture{};
    std::shared_ptr<resources::Shader> fxaaShader{nullptr};
};
}  // namespace spark::effects
