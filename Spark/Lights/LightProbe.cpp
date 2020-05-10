#include "LightProbe.h"

#include "CommonUtils.h"
#include "EngineSystems/SparkRenderer.h"
#include "Enums.h"
#include "GameObject.h"
#include "Mesh.h"
#include "ReflectionUtils.h"
#include "RenderingRequest.h"
#include "Shader.h"
#include "ShapeCreator.h"
#include "Structs.h"
#include "Texture.h"

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
    sphere->gpuLoad();
}

LightProbe::~LightProbe()
{
    glDeleteTextures(1, &prefilterCubemap);
    glDeleteTextures(1, &irradianceCubemap);
    irradianceCubemap = prefilterCubemap = 0;

    sphere->gpuUnload();
}

void LightProbe::update()
{
    if(!addedToLightManager)
    {
        SceneManager::getInstance()->getCurrentScene()->lightManager->addLightProbe(shared_from_base<LightProbe>());
        addedToLightManager = true;
    }

    glm::mat4 sphereModel(1);
    sphereModel = glm::scale(sphereModel, glm::vec3(radius));
    sphereModel[3] = glm::vec4(getGameObject()->transform.world.getPosition(), 1.0f);

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
        dirty = true;
    }

    float r = getRadius();
    ImGui::DragFloat("radius", &r, 0.1f);

    if(r < 0.0f)
        r = 0.0f;

    if(r != getRadius())
        setRadius(r);

    removeComponentGUI<LightProbe>();
}

LightProbeData LightProbe::getLightData() const
{
    LightProbeData data{};
    data.irradianceCubemapHandle = irradianceCubemapHandle;
    data.prefilterCubemapHandle = prefilterCubemapHandle;
    data.position = getGameObject()->transform.world.getPosition();
    data.radius = getRadius();
    return data;
}

bool LightProbe::getDirty() const
{
    return dirty;
}

float LightProbe::getRadius() const
{
    return radius;
}

void LightProbe::resetDirty()
{
    dirty = false;
}

GLuint LightProbe::getPrefilterCubemap() const
{
    return prefilterCubemap;
}

GLuint LightProbe::getIrradianceCubemap() const
{
    return irradianceCubemap;
}

void LightProbe::renderIntoIrradianceCubemap(GLuint framebuffer, GLuint environmentCubemap, Cube& cube, glm::mat4 projection,
                                             const std::array<glm::mat4, 6>& views, const std::shared_ptr<resources::Shader>& irradianceShader) const
{
    PUSH_DEBUG_GROUP(IRRADIANCE_CUBEMAP);

    glViewport(0, 0, irradianceCubemapSize, irradianceCubemapSize);

    irradianceShader->use();
    glBindTextureUnit(0, environmentCubemap);
    irradianceShader->setMat4("projection", projection);

    for(unsigned int i = 0; i < 6; ++i)
    {
        irradianceShader->setMat4("view", views[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceCubemap, 0);

        cube.draw();
    }

    POP_DEBUG_GROUP();
}

void LightProbe::renderIntoPrefilterCubemap(GLuint framebuffer, GLuint environmentCubemap, unsigned envCubemapSize, Cube& cube, glm::mat4 projection,
                                            const std::array<glm::mat4, 6>& views, const std::shared_ptr<resources::Shader>& prefilterShader) const
{
    PUSH_DEBUG_GROUP(PREFILTER_CUBEMAP);

    prefilterShader->use();
    glBindTextureUnit(0, environmentCubemap);
    prefilterShader->setMat4("projection", projection);
    prefilterShader->setFloat("textureSize", static_cast<float>(envCubemapSize));

    const GLuint maxMipLevels = 5;
    for(unsigned int mip = 0; mip < maxMipLevels; ++mip)
    {
        const auto mipWidth = static_cast<unsigned int>(prefilterCubemapSize * std::pow(0.5, mip));
        const auto mipHeight = static_cast<unsigned int>(prefilterCubemapSize * std::pow(0.5, mip));
        glViewport(0, 0, mipWidth, mipHeight);

        const float roughness = static_cast<float>(mip) / static_cast<float>(maxMipLevels - 1);
        prefilterShader->setFloat("roughness", roughness);
        for(unsigned int i = 0; i < 6; ++i)
        {
            prefilterShader->setMat4("view", views[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, prefilterCubemap, mip);

            cube.draw();
        }
    }

    POP_DEBUG_GROUP();
}

void LightProbe::setActive(bool active_)
{
    dirty = true;
    active = active_;
}

void LightProbe::setRadius(float radius_)
{
    dirty = true;
    radius = radius_;
}

// void LightProbe::setIrradianceCubemap(GLuint irradianceCubemap_)
//{
//    irradianceCubemap = irradianceCubemap_;
//}
//
// void LightProbe::setPrefilterCubemap(GLuint prefilterCubemap_)
//{
//    prefilterCubemap = prefilterCubemap_;
//}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::LightProbe>("LightProbe")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        //.property("dirty", &spark::DirectionalLight::dirty) //FIXME: shouldn't it always be dirty when loaded? maybe not
        //.property("addedToLightManager", &spark::DirectionalLight::addedToLightManager)
        .property("radius", &spark::LightProbe::radius)
        .property("generateLightProbe", &spark::LightProbe::generateLightProbe)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("irradianceCubemap", &spark::LightProbe::irradianceCubemap)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("prefilterCubemap", &spark::LightProbe::prefilterCubemap)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("addedToLightManager", &spark::LightProbe::addedToLightManager)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("irradianceCubemapHandle",
                  &spark::LightProbe::irradianceCubemapHandle)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("prefilterCubemapHandle",
                  &spark::LightProbe::prefilterCubemapHandle)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("dirty", &spark::LightProbe::dirty)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("irradianceCubemapSize",
                  &spark::LightProbe::irradianceCubemapSize)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("prefilterCubemapSize",
                  &spark::LightProbe::prefilterCubemapSize)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false));
}