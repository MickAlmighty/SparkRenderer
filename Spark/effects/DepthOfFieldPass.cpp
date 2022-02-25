#include "DepthOfFieldPass.h"

#include <array>

#include "utils/CommonUtils.h"
#include "ICamera.hpp"
#include "Shader.h"
#include "Spark.h"

namespace
{
template<size_t T>
void generatePoints(std::array<glm::vec2, T>& points, float angleOffsetRad = 0.0f)
{
    const float GOLDEN_ANGLE = 2.39996323f;

    auto divident = sqrt((float)T);
    for(int j = 0; j < T; ++j)
    {
        float theta = j * GOLDEN_ANGLE + angleOffsetRad;
        float r = sqrt((float)j) / divident;

        points[j] = glm::vec2(r * cos(theta), r * sin(theta));
    }
}
}  // namespace

namespace spark::effects
{
DepthOfFieldPass::DepthOfFieldPass(unsigned int width, unsigned int height) : w(width), h(height)
{
    cocShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/circleOfConfusion.glsl");
    blendShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/blendDof.glsl");
    poissonBlurShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/poissonBlur.glsl");
    poissonBlurShader2 = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/poissonBlur2.glsl");

    std::array<glm::vec2, 16> taps16Array;
    std::array<glm::vec2, 16> taps16Array2;

    generatePoints(taps16Array);
    generatePoints(taps16Array2, glm::radians(3.0f));

    taps16.updateData(taps16Array);
    taps16_2.updateData(taps16Array2);

    createFrameBuffersAndTextures();
}

GLuint DepthOfFieldPass::process(GLuint lightPassTexture, GLuint depthTexture, const std::shared_ptr<ICamera>& camera) const
{
    PUSH_DEBUG_GROUP(DEPTH_OF_FIELD)
    calculateCircleOfConfusion(lightPassTexture, depthTexture, camera);
    blurLightPassTexture(depthTexture);
    blendDepthOfField(lightPassTexture);

    POP_DEBUG_GROUP();
    return blendDofTexture.get();
}

void DepthOfFieldPass::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createFrameBuffersAndTextures();
}

void DepthOfFieldPass::createFrameBuffersAndTextures()
{
    poissonBlurTexture = utils::createTexture2D(w / 2, h / 2, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    poissonBlurTexture2 = utils::createTexture2D(w / 2, h / 2, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    cocTexture = utils::createTexture2D(w, h, GL_RGBA16F, GL_RGBA, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    blendDofTexture = utils::createTexture2D(w, h, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(poissonBlurFramebuffer, {poissonBlurTexture.get()});
    utils::recreateFramebuffer(poissonBlurFramebuffer2, {poissonBlurTexture2.get()});
    utils::recreateFramebuffer(cocFramebuffer, {cocTexture.get()});
    utils::recreateFramebuffer(blendDofFramebuffer, {blendDofTexture.get()});
}

DepthOfFieldPass::~DepthOfFieldPass()
{
    const GLuint framebuffers[] = {cocFramebuffer, blendDofFramebuffer, poissonBlurFramebuffer, poissonBlurFramebuffer2};
    glDeleteFramebuffers(4, framebuffers);
    cocFramebuffer = blendDofFramebuffer = 0;
}

void DepthOfFieldPass::calculateCircleOfConfusion(GLuint lightingTexture, GLuint depthTexture, const std::shared_ptr<ICamera>& camera) const
{
    PUSH_DEBUG_GROUP(COC_COMPUTING)

    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, cocFramebuffer);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    cocShader->use();
    cocShader->setFloat("u_Uniforms.A", aperture);
    cocShader->setFloat("u_Uniforms.f", f);
    cocShader->setFloat("u_Uniforms.S1", focusPoint);
    cocShader->setFloat("u_Uniforms.maxCoC", maxCoC);
    cocShader->bindUniformBuffer("Camera", camera->getUbo());
    glBindTextureUnit(0, depthTexture);
    glBindTextureUnit(1, lightingTexture);
    screenQuad.draw();
    glBindTextureUnit(0, 0);

    POP_DEBUG_GROUP();
}

void DepthOfFieldPass::blurLightPassTexture(GLuint depthTexture) const
{
    glViewport(0, 0, w / 2, h / 2);
    glBindFramebuffer(GL_FRAMEBUFFER, poissonBlurFramebuffer);
    poissonBlurShader->use();
    poissonBlurShader->setFloat("u_Uniforms.scale", poissonBlurScale);
    poissonBlurShader->bindSSBO("Taps", taps16);
    glBindTextureUnit(0, cocTexture.get());
    glBindTextureUnit(1, depthTexture);
    screenQuad.draw();
    glBindTextureUnit(1, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, poissonBlurFramebuffer2);
    poissonBlurShader2->use();
    poissonBlurShader2->setFloat("u_Uniforms.scale", poissonBlurScale * 0.5f);
    poissonBlurShader2->bindSSBO("Taps", taps16_2);
    glBindTextureUnit(0, poissonBlurTexture.get());
    screenQuad.draw();
    glBindTextureUnit(0, 0);

    glViewport(0, 0, w, h);
}

void DepthOfFieldPass::blendDepthOfField(GLuint lightPassTexture) const
{
    PUSH_DEBUG_GROUP(BLEND_DOF);
    glViewport(0, 0, w, h);

    glBindFramebuffer(GL_FRAMEBUFFER, blendDofFramebuffer);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    blendShader->use();
    blendShader->setFloat("u_Uniforms.maxCoC", maxCoC);
    GLuint textures[3] = {cocTexture.get(), lightPassTexture, poissonBlurTexture2.get()};
    glBindTextures(0, 3, textures);
    screenQuad.draw();
    glBindTextures(0, 3, nullptr);

    POP_DEBUG_GROUP();
}
}  // namespace spark::effects
