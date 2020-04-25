#include "LightProbe.h"

#include "GameObject.h"
#include "ReflectionUtils.h"

namespace spark
{
LightProbe::LightProbe() : Component("PointLight") {}

void LightProbe::update()
{
    if(!addedToLightManager)
    {
        SceneManager::getInstance()->getCurrentScene()->lightManager->addLightProbe(shared_from_base<LightProbe>());
        addedToLightManager = true;
    }
}

void LightProbe::fixedUpdate() {}

void spark::LightProbe::drawGUI()
{
    if (ImGui::Button("Generate Light Probe"))
    {
        generateLightProbe = true;
    }
    removeComponentGUI<LightProbe>();
}

GLuint LightProbe::getPrefilterCubemap() const
{
    return prefilterCubemap;
}

GLuint LightProbe::getIrradianceCubemap() const
{
    return irradianceCubemap;
}

void LightProbe::setIrradianceCubemap(GLuint irradianceCubemap_)
{
    irradianceCubemap = irradianceCubemap_;
}

void LightProbe::setPrefilterCubemap(GLuint prefilterCubemap_)
{
    prefilterCubemap = prefilterCubemap_;
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::LightProbe>("LightProbe")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        //.property("dirty", &spark::DirectionalLight::dirty) //FIXME: shouldn't it always be dirty when loaded? maybe not
        //.property("addedToLightManager", &spark::DirectionalLight::addedToLightManager)
        .property("generateLightProbe", &spark::LightProbe::generateLightProbe)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("irradianceCubemap", &spark::LightProbe::irradianceCubemap)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("prefilterCubemap", &spark::LightProbe::prefilterCubemap)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false))
        .property("addedToLightManager", &spark::LightProbe::addedToLightManager)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false));
}