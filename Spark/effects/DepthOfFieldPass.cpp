#include "DepthOfFieldPass.h"

#include "utils/CommonUtils.h"
#include "ICamera.hpp"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
DepthOfFieldPass::DepthOfFieldPass(unsigned int width, unsigned int height)
    : w(width), h(height), blurPass(w / 2, h / 2)
{
    cocShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/circleOfConfusion.glsl");
    blendShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/blendDof.glsl");

    createFrameBuffersAndTextures();
    // glGenBuffers(1, &indirectBufferID);
    // glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirectBufferID);
    // DrawArraysIndirectCommand indirectCmd{};
    // indirectCmd.count = 1;
    // indirectCmd.instanceCount = 0;
    // indirectCmd.first = 0;
    // indirectCmd.baseInstance = 0;
    // glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(DrawArraysIndirectCommand), &indirectCmd, GL_DYNAMIC_DRAW);

    //// Create a texture proxy for the indirect buffer
    //// (used during bokeh count synch.)
    // glGenTextures(1, &bokehCountTexID);
    // glBindTexture(GL_TEXTURE_BUFFER, bokehCountTexID);
    // glTexBuffer(GL_TEXTURE_BUFFER, GL_R32UI, indirectBufferID);
    //// Create an atomic counter
    // glGenBuffers(1, &bokehCounterID);
    // glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, bokehCounterID);
    // glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(unsigned int), 0, GL_DYNAMIC_DRAW);

    // glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, bokehCounterID);

    // bokehPositionBuffer.genBuffer(sizeof(glm::vec4) * 1024);
    // glGenTextures(1, &bokehPositionTexture);
    // glBindTexture(GL_TEXTURE_BUFFER, bokehPositionTexture);
    // glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, bokehPositionBuffer.ID);

    // bokehColorBuffer.genBuffer(sizeof(glm::vec4) * 1024);
    // glGenTextures(1, &bokehColorTexture);
    // glBindTexture(GL_TEXTURE_BUFFER, bokehColorTexture);
    // glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, bokehColorBuffer.ID);
}

GLuint DepthOfFieldPass::process(GLuint lightPassTexture, GLuint depthTexture, const std::shared_ptr<ICamera>& camera) const
{
    PUSH_DEBUG_GROUP(DEPTH_OF_FIELD)
    calculateCircleOfConfusion(depthTexture, camera);
    blurLightPassTexture(lightPassTexture);
    detectBokehPositions(lightPassTexture);
    renderBokehShapes();
    blendDepthOfField(lightPassTexture);

    /*glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, bokehCounterID);
    glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(unsigned int), 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bokehPositionBuffer.ID);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * 1024, 0, GL_DYNAMIC_DRAW);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bokehColorBuffer.ID);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::vec4) * 1024, 0, GL_DYNAMIC_DRAW);
    */
    POP_DEBUG_GROUP();
    return blendDofTexture.get();
}

void DepthOfFieldPass::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    blurPass.resize(width / 2, height / 2);
    createFrameBuffersAndTextures();
}

void DepthOfFieldPass::createFrameBuffersAndTextures()
{
    cocTexture = utils::createTexture2D(w, h, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    blendDofTexture = utils::createTexture2D(w, h, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(cocFramebuffer, {cocTexture.get()});
    utils::recreateFramebuffer(blendDofFramebuffer, {blendDofTexture.get()});
}

DepthOfFieldPass::~DepthOfFieldPass()
{
    const GLuint framebuffers[] = {cocFramebuffer, blendDofFramebuffer};
    glDeleteFramebuffers(2, framebuffers);
    cocFramebuffer = blendDofFramebuffer = 0;

    /*deleteFrameBuffersAndTextures();
    glDeleteTextures(1, &randomNormalsTexture);
    glDeleteTextures(1, &bokehCountTexID);
    glDeleteBuffers(1, &bokehCounterID);
    glDeleteBuffers(1, &indirectBufferID);
    bokehPositionBuffer.cleanup();
    bokehColorBuffer.cleanup();
    glDeleteTextures(1, &bokehPositionTexture);
    glDeleteTextures(1, &bokehColorTexture);*/
}

void DepthOfFieldPass::calculateCircleOfConfusion(GLuint depthTexture, const std::shared_ptr<ICamera>& camera) const
{
    PUSH_DEBUG_GROUP(COC_COMPUTING)

    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, cocFramebuffer);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);

    cocShader->use();
    cocShader->setFloat("zNear", nearStart);
    cocShader->setFloat("zNearEnd", nearEnd);
    cocShader->setFloat("zFarStart", farStart);
    cocShader->setFloat("zFar", farEnd);
    cocShader->bindUniformBuffer("Camera.camera", camera->getUbo());
    glBindTextureUnit(0, depthTexture);
    screenQuad.draw();
    glBindTextureUnit(0, 0);

    POP_DEBUG_GROUP();
}

void DepthOfFieldPass::blurLightPassTexture(GLuint lightPassTexture) const
{
    blurPass.blurTexture(lightPassTexture);
    blurPass.blurTexture(blurPass.getBlurredTexture());
    blurPass.blurTexture(blurPass.getBlurredTexture());
}

void DepthOfFieldPass::detectBokehPositions(GLuint lightPassTexture) const
{
    // const auto bokehDetectionShader = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Shader>("shaders/bokehDetection.glsl");
    // bokehDetectionShader->use();
    // GLuint textures[2] = {lightPassTexture, circleOfConfusionTexture};
    // glBindTextures(3, 2, textures);

    // glActiveTexture(GL_TEXTURE0 + 1);
    // glBindImageTexture(1, bokehPositionTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA32F);
    //// Bind color image buffer
    // glActiveTexture(GL_TEXTURE0 + 2);
    // glBindImageTexture(2, bokehColorTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA32F);

    // screenQuad.draw();
    // glBindTextures(3, 2, nullptr);
}

void DepthOfFieldPass::renderBokehShapes() const {}

void DepthOfFieldPass::blendDepthOfField(GLuint lightPassTexture) const
{
    PUSH_DEBUG_GROUP(BLEND_DOF);
    glViewport(0, 0, w, h);

    glBindFramebuffer(GL_FRAMEBUFFER, blendDofFramebuffer);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    blendShader->use();

    GLuint textures[3] = {cocTexture.get(), lightPassTexture, blurPass.getBlurredTexture()};
    glBindTextures(0, 3, textures);
    screenQuad.draw();
    glBindTextures(0, 3, nullptr);

    POP_DEBUG_GROUP();
}
}  // namespace spark::effects
