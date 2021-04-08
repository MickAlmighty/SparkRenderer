#include "LightProbe.h"

#include <glm/gtx/transform.hpp>

#include "CommonUtils.h"
#include "EngineSystems/SparkRenderer.h"
#include "Enums.h"
#include "GameObject.h"
#include "Mesh.h"
#include "RenderingRequest.h"
#include "ResourceLibrary.h"
#include "Shader.h"
#include "ShapeCreator.h"
#include "Spark.h"
#include "Structs.h"
#include "Texture.h"
#include "Timer.h"

namespace spark
{
LightProbe::LightProbe() : Component("LightProbe")
{
    utils::createCubemap(prefilterCubemap, prefilterCubemapSize, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, true);
    prefilterCubemapHandle = glGetTextureHandleARB(prefilterCubemap);
    glMakeTextureHandleResidentARB(prefilterCubemapHandle);

    utils::createCubemap(irradianceCubemap, irradianceCubemapSize, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    irradianceCubemapHandle = glGetTextureHandleARB(irradianceCubemap);
    glMakeTextureHandleResidentARB(irradianceCubemapHandle);

    const auto attribute = VertexShaderAttribute::createVertexShaderAttributeInfo(0, 3, ShapeCreator::createSphere(1.0f, 10));
    sphere = std::make_shared<Mesh>(std::vector<VertexShaderAttribute>{attribute}, std::vector<unsigned int>{},
                                    std::map<TextureTarget, std::shared_ptr<resources::Texture>>{}, "Mesh", ShaderType::SOLID_COLOR_SHADER);
}

LightProbe::~LightProbe()
{
    glDeleteTextures(1, &prefilterCubemap);
    glDeleteTextures(1, &irradianceCubemap);
    irradianceCubemap = prefilterCubemap = 0;

    notifyAbout(LightCommand::remove);
}

void LightProbe::update()
{
    if(!lightManager)
    {
        lightManager = SceneManager::getInstance()->getCurrentScene()->lightManager;
        add(lightManager);

        notifyAbout(LightCommand::add);
    }

    const glm::vec3 gameObjPosition = getGameObject()->transform.world.getPosition();
    if(position != gameObjPosition)
    {
        position = gameObjPosition;
        notifyAbout(LightCommand::update);
    }

    glm::mat4 sphereModel(1);
    sphereModel = glm::scale(sphereModel, glm::vec3(radius));
    sphereModel[3] = glm::vec4(position, 1.0f);

    if(getGameObject() == getGameObject()->getScene()->getGameObjectToPreview())
    {
        if(sphere)
        {
            RenderingRequest request{};
            request.shaderType = sphere->shaderType;
            request.gameObject = getGameObject();
            request.mesh = sphere;
            request.model = sphereModel;

            SparkRenderer::getInstance()->addRenderingRequest(request);
        }
    }
}

void LightProbe::fixedUpdate() {}

void LightProbe::drawGUI()
{
    if(ImGui::Button("Generate Light Probe"))
    {
        generateLightProbe = true;
    }

    float r = getRadius();
    float fDist = getFadeDistance();
    ImGui::DragFloat("radius", &r, 0.1f);
    ImGui::DragFloat("fadeDistance", &fDist, 0.1f);

    if(r < 0.0f)
        r = 0.0f;

    if(r != getRadius())
        setRadius(r);

    if(fDist < 0.0f)
        fDist = 0.0f;

    if(fDist != getFadeDistance())
        setFadeDistance(fDist);

    removeComponentGUI<LightProbe>();
}

LightProbeData LightProbe::getLightData() const
{
    LightProbeData data{};
    data.irradianceCubemapHandle = irradianceCubemapHandle;
    data.prefilterCubemapHandle = prefilterCubemapHandle;
    data.positionAndRadius = glm::vec4(position, getRadius());
    data.fadeDistance = getFadeDistance();
    return data;
}

float LightProbe::getRadius() const
{
    return radius;
}

float LightProbe::getFadeDistance() const
{
    return fadeDistance;
}

GLuint LightProbe::getPrefilterCubemap() const
{
    return prefilterCubemap;
}

GLuint LightProbe::getIrradianceCubemap() const
{
    return irradianceCubemap;
}

void LightProbe::renderIntoIrradianceCubemap(GLuint framebuffer, GLuint environmentCubemap, Cube& cube,
                                             const std::shared_ptr<resources::Shader>& irradianceShader) const
{
    PUSH_DEBUG_GROUP(IRRADIANCE_CUBEMAP)

    glViewport(0, 0, irradianceCubemapSize, irradianceCubemapSize);

    irradianceShader->use();
    glBindTextureUnit(0, environmentCubemap);

    glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, irradianceCubemap, 0);
    cube.draw();

    glFinish();
    POP_DEBUG_GROUP();
}

void LightProbe::renderIntoPrefilterCubemap(GLuint framebuffer, GLuint environmentCubemap, unsigned envCubemapSize, Cube& cube,
                                            const std::shared_ptr<resources::Shader>& prefilterShader,
                                            const std::shared_ptr<resources::Shader>& resampleCubemapShader) const
{
    PUSH_DEBUG_GROUP(PREFILTER_CUBEMAP);

    {
        resampleCubemapShader->use();
        glBindTextureUnit(0, environmentCubemap);

        glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, prefilterCubemap, 0);

        glViewport(0, 0, prefilterCubemapSize, prefilterCubemapSize);
        cube.draw();
    }

    const GLuint maxMipLevels = 5;
    prefilterShader->use();
    glBindTextureUnit(0, environmentCubemap);
    prefilterShader->setFloat("textureSize", static_cast<float>(envCubemapSize));

    for(unsigned int mip = 0; mip < maxMipLevels; ++mip)
    {
        const auto mipSize = static_cast<unsigned int>(prefilterCubemapSize * std::pow(0.5, mip));
        glViewport(0, 0, mipSize, mipSize);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, prefilterCubemap, mip);

        const float roughness = static_cast<float>(mip) / static_cast<float>(maxMipLevels - 1);
        prefilterShader->setFloat("roughness", roughness);

        cube.draw();
    }

    glFinish();
    POP_DEBUG_GROUP();
}

void LightProbe::setActive(bool active_)
{
    active = active_;
    if(active)
    {
        notifyAbout(LightCommand::add);
    }
    else
    {
        notifyAbout(LightCommand::remove);
    }
}

void LightProbe::setRadius(float radius_)
{
    radius = radius_;
    notifyAbout(LightCommand::update);
}

void LightProbe::setFadeDistance(float fadeDistance_)
{
    fadeDistance = fadeDistance_;
    notifyAbout(LightCommand::update);
}

void LightProbe::notifyAbout(LightCommand command)
{
    const LightStatus<LightProbe> status{command, this};
    notify(&status);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::LightProbe>("LightProbe")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("radius", &spark::LightProbe::radius)
        .property("fadeDistance", &spark::LightProbe::fadeDistance);
}