#include "LightManager.h"

#include "DirectionalLight.h"
#include "LightProbe.h"
#include "PointLight.h"
#include "SpotLight.h"

namespace spark::lights
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
        if(const auto lightData = prepareLightDataBuffer<DirectionalLightData>(directionalLights); lightData.empty())
        {
            dirLightSSBO.clearData();
        }
        else
        {
            dirLightSSBO.updateData(lightData);
        }
        areDirLightsDirty = false;
    }
}

void LightManager::updatePointLightBuffer()
{
    if(arePointLightsDirty)
    {
        if(const auto lightData = prepareLightDataBuffer<PointLightData>(pointLights); lightData.empty())
        {
            pointLightSSBO.clearData();
        }
        else
        {
            pointLightSSBO.updateData(lightData);
        }
        arePointLightsDirty = false;
    }
}

void LightManager::updateSpotLightBuffer()
{
    if(areSpotLightsDirty)
    {
        if(const auto lightData = prepareLightDataBuffer<SpotLightData>(spotLights); lightData.empty())
        {
            spotLightSSBO.clearData();
        }
        else
        {
            spotLightSSBO.updateData(lightData);
        }
        areSpotLightsDirty = false;
    }
}

void LightManager::updateLightProbeBuffer()
{
    if(areLightProbesDirty)
    {
        if(auto lightDataBuffer = prepareLightDataBuffer<LightProbeData>(lightProbes); lightDataBuffer.empty())
        {
            lightProbeSSBO.clearData();
        }
        else
        {
            std::sort(lightDataBuffer.begin(), lightDataBuffer.end());
            lightProbeSSBO.updateData(lightDataBuffer);
        }
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

}  // namespace spark::lights