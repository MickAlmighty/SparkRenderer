#include "LightManager.h"

#include "DirectionalLight.h"
#include "GameObject.h"
#include "LightProbe.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "Structs.h"

namespace spark
{
void LightManager::updateLightBuffers()
{
    updateDirLightBuffer();
    updatePointLightBuffer();
    updateSpotLightBuffer();
    updateLightProbeBuffer();
}

const SSBO& LightManager::getDirLightSSBO() const
{
    return dirLightSSBO;
}

const SSBO& LightManager::getPointLightSSBO() const
{
    return pointLightSSBO;
}

const SSBO& LightManager::getSpotLightSSBO() const
{
    return spotLightSSBO;
}

const SSBO& LightManager::getLightProbeSSBO() const
{
    return lightProbeSSBO;
}

const std::vector<DirectionalLight*>& LightManager::getDirLights() const
{
    return directionalLights;
}

const std::vector<PointLight*>& LightManager::getPointLights() const
{
    return pointLights;
}

const std::vector<SpotLight*>& LightManager::getSpotLights() const
{
    return spotLights;
}

const std::vector<LightProbe*>& LightManager::getLightProbes() const
{
    return lightProbes;
}

void LightManager::updateDirLightBuffer()
{
    if(areDirLightsDirty)
    {
        dirLightSSBO.updateData(prepareLightDataBuffer<DirectionalLightData>(directionalLights));
        areDirLightsDirty = false;
    }
}

void LightManager::updatePointLightBuffer()
{
    if(arePointLightsDirty)
    {
        pointLightSSBO.updateData(prepareLightDataBuffer<PointLightData>(pointLights));
        arePointLightsDirty = false;
    }
}

void LightManager::updateSpotLightBuffer()
{
    if(areSpotLightsDirty)
    {
        spotLightSSBO.updateData(prepareLightDataBuffer<SpotLightData>(spotLights));
        areSpotLightsDirty = false;
    }
}

void LightManager::updateLightProbeBuffer()
{
    if(areLightProbesDirty)
    {
        auto lightDataBuffer = prepareLightDataBuffer<LightProbeData>(lightProbes);
        std::sort(lightDataBuffer.begin(), lightDataBuffer.end());
        lightProbeSSBO.updateData(lightDataBuffer);
        areLightProbesDirty = false;
    }
}

void LightManager::update(const LightStatus<DirectionalLight>* const dirLightStatus)
{
    processLightStatus(dirLightStatus, areDirLightsDirty, directionalLights);
}

void LightManager::update(const LightStatus<PointLight>* const pointLightStatus)
{
    processLightStatus(pointLightStatus, arePointLightsDirty, pointLights);
}

void LightManager::update(const LightStatus<SpotLight>* const spotLightStatus)
{
    processLightStatus(spotLightStatus, areSpotLightsDirty, spotLights);
}

void LightManager::update(const LightStatus<LightProbe>* const lightProbeStatus)
{
    processLightStatus(lightProbeStatus, areLightProbesDirty, lightProbes);
}

}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::LightManager>("LightManager").constructor()(rttr::policy::ctor::as_std_shared_ptr);
}