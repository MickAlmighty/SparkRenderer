#include "LightManager.h"

#include "DirectionalLight.h"
#include "LightProbe.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "Structs.h"

namespace spark
{
void LightManager::addLightProbe(const std::shared_ptr<LightProbe>& lightProbe)
{
    lightProbes.push_back(lightProbe);
}

void LightManager::addDirectionalLight(const std::shared_ptr<DirectionalLight>& directionalLight)
{
    directionalLights.push_back(directionalLight);
}

void LightManager::addPointLight(const std::shared_ptr<PointLight>& pointLight)
{
    pointLights.push_back(pointLight);
}

void LightManager::addSpotLight(const std::shared_ptr<SpotLight>& spotLight)
{
    spotLights.push_back(spotLight);
}

void LightManager::updateLightBuffers()
{
    const auto dirLightDataBuffer = getLightDataBuffer<DirectionalLightData, DirectionalLight>(directionalLights);
    updateBufferIfNecessary(dirLightDataBuffer, dirLightSSBO);

    const auto pointLightDataBuffer = getLightDataBuffer<PointLightData, PointLight>(pointLights);
    updateBufferIfNecessary(pointLightDataBuffer, pointLightSSBO);

    const auto spotLightDataBuffer = getLightDataBuffer<SpotLightData, SpotLight>(spotLights);
    updateBufferIfNecessary(spotLightDataBuffer, spotLightSSBO);

    const auto lightProbesDataBuffer = getLightDataBuffer<LightProbeData, LightProbe>(lightProbes);
    updateBufferIfNecessary(lightProbesDataBuffer, lightProbeSSBO);
}

LightManager::LightManager()
{
    dirLightSSBO.genBuffer();
    pointLightSSBO.genBuffer();
    spotLightSSBO.genBuffer();
    lightProbeSSBO.genBuffer();
}

LightManager::~LightManager()
{
    dirLightSSBO.cleanup();
    spotLightSSBO.cleanup();
    pointLightSSBO.cleanup();
    lightProbeSSBO.cleanup();
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::LightManager>("LightManager").constructor()(rttr::policy::ctor::as_std_shared_ptr);
}