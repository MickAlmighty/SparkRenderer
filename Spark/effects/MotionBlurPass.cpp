#include "MotionBlurPass.hpp"

#include "Clock.h"
#include "utils/CommonUtils.h"
#include "ICamera.hpp"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
MotionBlurPass::MotionBlurPass(unsigned int width, unsigned int height) : w(width), h(height)
{
    motionBlurShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/motionBlur.glsl");
    createFrameBuffersAndTextures();
}

MotionBlurPass::~MotionBlurPass()
{
    glDeleteFramebuffers(1, &framebuffer1);
}

std::optional<GLuint> MotionBlurPass::process(const std::shared_ptr<ICamera>& camera, GLuint colorTexture, GLuint depthTexture)
{
    const glm::f64mat4 viewMatrix = camera->getViewMatrix();
    const glm::f64mat4 projectionMatrix = camera->getProjectionReversedZ();
    const glm::f64mat4 projectionView = projectionMatrix * viewMatrix;

    if(!initialized)
    {
        // it is necessary when the scene has been loaded and
        // the difference between current VP and last frame VP matrices generates huge velocities for all pixels
        // so it needs to be reset
        prevProjectionView = projectionView;
        initialized = true;
    }

    if(projectionView == prevProjectionView)
        return {};

    const auto blurScale = []
    {
        constexpr double deltaTarget = 1.0 / 60.0;
        if(const auto blurScaleBase = static_cast<float>(deltaTarget / Clock::getDeltaTime()); blurScaleBase < 1.0f)
        {
            return blurScaleBase;
        }
        else
        {
            return 1.0f + glm::log2(blurScaleBase);
        }
    }();

    PUSH_DEBUG_GROUP(MOTION_BLUR)
    {
        glViewport(0, 0, w, h);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer1);

        motionBlurShader->use();
        motionBlurShader->bindUniformBuffer("Camera.camera", camera->getUbo());
        motionBlurShader->setMat4("prevViewProj", prevProjectionView);

        motionBlurShader->setFloat("blurScale", blurScale);

        const glm::vec2 texelSize = {1.0f / static_cast<float>(w), 1.0f / static_cast<float>(h)};
        motionBlurShader->setVec2("texelSize", texelSize);

        const std::array<GLuint, 2> textures{colorTexture, depthTexture};
        glBindTextures(0, textures.size(), textures.data());
        screenQuad.draw();
    }

    glBindTextures(0, 2, nullptr);
    prevProjectionView = projectionView;

    POP_DEBUG_GROUP()
    return texture1.get();
}

void MotionBlurPass::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createFrameBuffersAndTextures();
}

void MotionBlurPass::createFrameBuffersAndTextures()
{
    texture1 = utils::createTexture2D(w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(framebuffer1, {texture1.get()});
}
}  // namespace spark::effects