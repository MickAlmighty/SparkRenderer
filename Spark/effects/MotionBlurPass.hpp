#pragma once

#include <optional>

#include "Buffer.hpp"
#include "Camera.h"
#include "ScreenQuad.hpp"
#include "Shader.h"

namespace spark::effects
{
class MotionBlurPass
{
    public:
    MotionBlurPass(unsigned int width, unsigned int height);
    MotionBlurPass(const MotionBlurPass&) = delete;
    MotionBlurPass(MotionBlurPass&&) = delete;
    MotionBlurPass& operator=(const MotionBlurPass&) = delete;
    MotionBlurPass& operator=(MotionBlurPass&&) = delete;
    ~MotionBlurPass();

    std::optional<GLuint> process(const std::shared_ptr<Camera>& camera, GLuint colorTexture, GLuint depthTexture);
    void resize(unsigned int width, unsigned int height);

    private:
    void createFrameBuffersAndTextures();

    unsigned int w{}, h{};
    ScreenQuad screenQuad;
    GLuint framebuffer1{};
    GLuint texture1{};
    std::shared_ptr<resources::Shader> motionBlurShader{nullptr};
};
}  // namespace spark::effects