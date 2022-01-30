#include "LightProbe.h"

#include <glm/gtx/transform.hpp>

#include "utils/CommonUtils.h"
#include "Enums.h"
#include "GUI/ImGui/imgui.h"
#include "GameObject.h"
#include "Mesh.h"
#include "renderers/RenderingRequest.h"
#include "ResourceLibrary.h"
#include "Scene.h"
#include "Shader.h"
#include "ShapeCreator.h"
#include "Texture.h"

namespace spark::lights
{
LightProbe::LightProbe() : Component()
{
    prefilterCubemap = utils::createCubemap(prefilterCubemapSize, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, true);
    prefilterCubemapHandle = glGetTextureHandleARB(prefilterCubemap.get());
    glMakeTextureHandleResidentARB(prefilterCubemapHandle);

    irradianceCubemap = utils::createCubemap(irradianceCubemapSize, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    irradianceCubemapHandle = glGetTextureHandleARB(irradianceCubemap.get());
    glMakeTextureHandleResidentARB(irradianceCubemapHandle);

    const auto attribute = VertexAttribute(0, 3, ShapeCreator::createSphere(1.0f, 10));
    auto vertexAttributes = std::vector{attribute};
    auto indices = std::vector<unsigned int>{};
    auto textures = std::map<TextureTarget, std::shared_ptr<resources::Texture>>{};
    sphere = std::make_shared<Mesh>(vertexAttributes, indices, textures, "Mesh", ShaderType::COLOR_ONLY);
}

LightProbe::~LightProbe()
{
    notifyAbout(LightCommand::remove);
}

void LightProbe::update()
{
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
            renderers::RenderingRequest request{};
            request.shaderType = sphere->shaderType;
            request.gameObject = getGameObject();
            request.mesh = sphere;
            request.model = sphereModel;

            getGameObject()->getScene()->addRenderingRequest(request);
        }
    }
}

void LightProbe::drawUIBody()
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
}

void LightProbe::start()
{
    add(getGameObject()->getScene()->lightManager);
    notifyAbout(LightCommand::add);
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
    return prefilterCubemap.get();
}

GLuint LightProbe::getIrradianceCubemap() const
{
    return irradianceCubemap.get();
}

void LightProbe::renderIntoIrradianceCubemap(GLuint framebuffer, GLuint environmentCubemap, Cube& cube,
                                             const std::shared_ptr<resources::Shader>& irradianceShader) const
{
    PUSH_DEBUG_GROUP(IRRADIANCE_CUBEMAP)

    glViewport(0, 0, irradianceCubemapSize, irradianceCubemapSize);

    irradianceShader->use();
    glBindTextureUnit(0, environmentCubemap);

    glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, irradianceCubemap.get(), 0);
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

        glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, prefilterCubemap.get(), 0);

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
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, prefilterCubemap.get(), mip);

        const float roughness = static_cast<float>(mip) / static_cast<float>(maxMipLevels - 1);
        prefilterShader->setFloat("roughness", roughness);

        cube.draw();
    }

    glFinish();
    POP_DEBUG_GROUP();
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

void LightProbe::onActive()
{
    notifyAbout(LightCommand::add);
}

void LightProbe::onInactive()
{
    notifyAbout(LightCommand::remove);
}

void LightProbe::notifyAbout(LightCommand command)
{
    const LightStatus<LightProbe> status{command, this};
    notify(&status);
}
}  // namespace spark::lights

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::lights::LightProbe>("LightProbe")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("radius", &spark::lights::LightProbe::getRadius, &spark::lights::LightProbe::setRadius)
        .property("fadeDistance", &spark::lights::LightProbe::getFadeDistance, &spark::lights::LightProbe::setFadeDistance);
}