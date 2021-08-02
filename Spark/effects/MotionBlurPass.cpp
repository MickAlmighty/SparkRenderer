#include "MotionBlurPass.hpp"

#include "Clock.h"
#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
MotionBlurPass::MotionBlurPass(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo) : w(width), h(height)
{
    motionBlurShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("motionBlur.glsl");
    motionBlurShader->bindUniformBuffer("Camera", cameraUbo);
    createFrameBuffersAndTextures();
}

MotionBlurPass::~MotionBlurPass()
{
    glDeleteTextures(1, &texture1);
    glDeleteFramebuffers(1, &framebuffer1);
}

std::optional<GLuint> MotionBlurPass::process(const std::shared_ptr<Camera>& camera, GLuint colorTexture, GLuint depthTexture)
{
    const glm::mat4 projectionView = camera->getProjectionReversedZInfiniteFarPlane() * camera->getViewMatrix();
    static glm::mat4 prevProjectionView = projectionView;

    if(projectionView == prevProjectionView)
        return {};

    if(static bool initialized = false; !initialized)
    {
        // it is necessary when the scene has been loaded and
        // the difference between current VP and last frame VP matrices generates huge velocities for all pixels
        // so it needs to be reset
        prevProjectionView = projectionView;
        initialized = true;
    }

    PUSH_DEBUG_GROUP(MOTION_BLUR);
    {
        glViewport(0, 0, w, h);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer1);

        motionBlurShader->use();
        motionBlurShader->setMat4("prevViewProj", prevProjectionView);
        motionBlurShader->setFloat("blurScale", static_cast<float>(Clock::getFPS()) / 60.0f);

        const glm::vec2 texelSize = {1.0f / static_cast<float>(w), 1.0f / static_cast<float>(h)};
        motionBlurShader->setVec2("texelSize", texelSize);

        const std::array<GLuint, 2> textures{colorTexture, depthTexture};
        glBindTextures(0, textures.size(), textures.data());
        screenQuad.draw();
    }

    glBindTextures(0, 2, nullptr);
    prevProjectionView = projectionView;

    POP_DEBUG_GROUP();
    return texture1;
}

void MotionBlurPass::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createFrameBuffersAndTextures();
}

void MotionBlurPass::createFrameBuffersAndTextures()
{
    utils::recreateTexture2D(texture1, w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(framebuffer1, {texture1});
}
}  // namespace spark::effects