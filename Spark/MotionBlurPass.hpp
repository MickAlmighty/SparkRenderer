#pragma once

#include "Camera.h"
#include "ScreenQuad.hpp"

namespace spark
{
class MotionBlurPass
{
    public:
    MotionBlurPass() = default;
    MotionBlurPass(const MotionBlurPass&) = delete;
    MotionBlurPass(MotionBlurPass&&) = delete;
    MotionBlurPass& operator=(const MotionBlurPass&) = delete;
    MotionBlurPass& operator=(MotionBlurPass&&) = delete;
    ~MotionBlurPass();

    void setup(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo);
    std::optional<GLuint> process(const std::shared_ptr<Camera>& camera, GLuint colorTexture, GLuint depthTexture);
    void createFrameBuffersAndTextures(unsigned int width, unsigned int height);
    void cleanup();

    private:
    unsigned int w, h;
    ScreenQuad screenQuad;
    GLuint framebuffer1{};
    GLuint texture1{};
    std::shared_ptr<resources::Shader> motionBlurShader{ nullptr };
};
}  // namespace spark