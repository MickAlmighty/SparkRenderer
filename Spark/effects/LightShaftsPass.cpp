#include "LightShaftsPass.hpp"

#include "CommonUtils.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::effects
{
LightShaftsPass::~LightShaftsPass()
{
    cleanup();
}

void LightShaftsPass::setup(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    lightShaftsShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("lightShafts.glsl");
    blendingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("renderTexture.glsl");
    blurPass = std::make_unique<BlurPass>(w / 4, h / 4);
}

std::optional<GLuint> LightShaftsPass::process(const std::shared_ptr<Camera>& camera, GLuint depthTexture, GLuint lightingTexture)
{
    const auto* const dirLight = lights::DirectionalLight::getDirLightForLightShafts();
    if(dirLight == nullptr)
        return {};

    const glm::vec2 lightScreenPos = dirLightPositionInScreenSpace(camera, dirLight);
    if(isCameraFacingDirectionalLight(lightScreenPos, camera, dirLight))
    {
        return {};
    }

    PUSH_DEBUG_GROUP(LIGHT SHAFTS);
    renderLightShaftsToTexture(dirLight, depthTexture, lightingTexture, lightScreenPos);
    blurLightShafts();
    blendLightShafts(lightingTexture);

    POP_DEBUG_GROUP();
    return blendingOutputTexture;
}

void LightShaftsPass::cleanup()
{
    const GLuint textures[3] = {radialBlurTexture1, radialBlurTexture2, blendingOutputTexture};
    glDeleteTextures(3, textures);
    const GLuint framebuffers[2] = {radialBlurFramebuffer1, blendingFramebuffer};
    glDeleteFramebuffers(2, framebuffers);
}

glm::vec2 LightShaftsPass::dirLightPositionInScreenSpace(const std::shared_ptr<Camera>& camera, const lights::DirectionalLight* const dirLight)
{
    const glm::mat4 view = camera->getViewMatrix();
    const glm::mat4 projection = camera->getProjectionReversedZ();

    const glm::vec3 camPos = camera->getPosition();

    const glm::vec3 dirLightPosition = dirLight->getDirection() * -glm::vec3(100);

    glm::vec4 dirLightNDCpos = projection * view * glm::vec4(dirLightPosition, 1.0f);
    dirLightNDCpos /= dirLightNDCpos.w;

    return glm::vec2((dirLightNDCpos.x + 1.0f) * 0.5f, (dirLightNDCpos.y + 1.0f) * 0.5f);
}

bool LightShaftsPass::isCameraFacingDirectionalLight(glm::vec2 dirLightScreenSpacePosition, const std::shared_ptr<Camera>& camera,
                                                     const lights::DirectionalLight* const dirLight)
{
    const bool isCameraFacingDirLight = glm::dot(dirLight->getDirection(), camera->getFront()) < 0.0f;
    const float distance = glm::distance(glm::vec2(0.5f), dirLightScreenSpacePosition);
    return distance > 1.0f || !isCameraFacingDirLight;
}

void LightShaftsPass::renderLightShaftsToTexture(const lights::DirectionalLight* const dirLight, GLuint depthTexture, GLuint lightingTexture,
                                                 const glm::vec2 lightScreenPos) const
{
    PUSH_DEBUG_GROUP(RADIAL BLUR);

    lightShaftsShader->use();
    lightShaftsShader->setVec2("lightScreenPos", lightScreenPos);
    lightShaftsShader->setVec3("lightColor", dirLight->getColor());
    lightShaftsShader->setFloat("exposure", exposure);
    lightShaftsShader->setFloat("decay", decay);
    lightShaftsShader->setFloat("density", density);
    lightShaftsShader->setFloat("weight", weight);

    glViewport(0, 0, w / 4, h / 4);

    glBindFramebuffer(GL_FRAMEBUFFER, radialBlurFramebuffer1);
    utils::bindTexture2D(radialBlurFramebuffer1, radialBlurTexture1);
    glBindTextureUnit(0, depthTexture);
    glBindTextureUnit(1, lightingTexture);
    screenQuad.draw();

    utils::bindTexture2D(radialBlurFramebuffer1, radialBlurTexture2);
    glBindTextureUnit(1, radialBlurTexture1);
    screenQuad.draw();

    glBindTextureUnit(0, 0);
    POP_DEBUG_GROUP()
}

void LightShaftsPass::blurLightShafts() const
{
    blurPass->blurTexture(radialBlurTexture2);
}

void LightShaftsPass::blendLightShafts(GLuint lightingTexture) const
{
    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, blendingFramebuffer);

    blendingShader->use();
    glBindTextureUnit(0, lightingTexture);
    screenQuad.draw();

    glBlendFunc(GL_ONE, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    glEnable(GL_BLEND);

    glBindTextureUnit(0, blurPass->getBlurredTexture());
    screenQuad.draw();
    glBindTextureUnit(0, 0);

    glDisable(GL_BLEND);
}

void LightShaftsPass::createFrameBuffersAndTextures(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    utils::recreateTexture2D(radialBlurTexture1, w / 4, h / 4, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(radialBlurTexture2, w / 4, h / 4, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(blendingOutputTexture, w, h, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateFramebuffer(radialBlurFramebuffer1, {radialBlurTexture1});
    utils::recreateFramebuffer(blendingFramebuffer, {blendingOutputTexture});
    blurPass->recreateWithNewSize(w / 4, h / 4);
}
}  // namespace spark::effects