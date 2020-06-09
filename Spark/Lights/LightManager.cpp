#include "LightManager.h"

#include "DirectionalLight.h"
#include "LightProbe.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "Structs.h"

namespace spark
{
bool LightManager::compareLightProbes::operator()(const std::weak_ptr<LightProbe>& l, const std::weak_ptr<LightProbe>& r) const 
{
    return l.lock()->getRadius() < r.lock()->getRadius();
}

void LightManager::addLightProbe(const std::shared_ptr<LightProbe>& lightProbe)
{
    lightProbes.insert(lightProbe);
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
    const auto dirLightDataBufferOpt = getLightDataBuffer<DirectionalLightData, DirectionalLight>(directionalLights);
    updateBufferIfNecessary(dirLightDataBufferOpt, dirLightSSBO);

    const auto pointLightDataBufferOpt = getLightDataBuffer<PointLightData, PointLight>(pointLights);
    updateBufferIfNecessary(pointLightDataBufferOpt, pointLightSSBO);

    const auto spotLightDataBufferOpt = getLightDataBuffer<SpotLightData, SpotLight>(spotLights);
    updateBufferIfNecessary(spotLightDataBufferOpt, spotLightSSBO);

    const auto lightProbesDataBufferOpt = getLightProbeDataBufer(lightProbes);
    updateBufferIfNecessary(lightProbesDataBufferOpt, lightProbeSSBO);
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

bool LightManager::removeExpiredLightPointers(std::multiset<std::weak_ptr<LightProbe>, compareLightProbes>& lightContainer)
{
    bool isBufferDirty = false;
    bool atLeastOnExpiredPointerRemoved = false;

    do
    {
        const auto containerIt = std::find_if(std::begin(lightContainer), std::end(lightContainer),
                                              [](const std::weak_ptr<LightProbe>& weakPtr) { return weakPtr.expired(); });

        if(containerIt != std::end(lightContainer))
        {
            lightContainer.erase(containerIt);
            atLeastOnExpiredPointerRemoved = true;
            isBufferDirty = true;
        }
        else
        {
            break;
        }
    } while(atLeastOnExpiredPointerRemoved);

    return isBufferDirty;
}

std::optional<std::vector<LightProbeData>> LightManager::getLightProbeDataBufer(
    std::multiset<std::weak_ptr<LightProbe>, compareLightProbes>& lightProbeContainer)
{
    bool isBufferDirty = removeExpiredLightPointers(lightProbeContainer);
    bool isAtLeastOneLightDirty = std::any_of(lightProbeContainer.begin(), lightProbeContainer.end(),
                                              [](const std::weak_ptr<LightProbe>& light) { return light.lock()->getDirty(); });

    if(isBufferDirty || isAtLeastOneLightDirty)
    {
        auto lightProbeIt = lightProbeContainer.begin();
        while(lightProbeIt != lightProbeContainer.end())
        {
            if(lightProbeIt->lock()->getDirty())
            {
                const auto lightProbeWeakPtr = *lightProbeIt;
                const auto lightProbeToEraseIt = lightProbeIt;
                lightProbeIt = std::next(lightProbeIt);
                lightProbeContainer.erase(lightProbeToEraseIt);

                // doing insertion sort here using the set container property
                lightProbeContainer.insert(lightProbeWeakPtr);
                lightProbeWeakPtr.lock()->resetDirty();
            }
            else
            {
                lightProbeIt = std::next(lightProbeIt);
            }
        }

        std::vector<LightProbeData> bufferData;
        bufferData.reserve(lightProbeContainer.size());

        for(const auto& light : lightProbeContainer)
        {
            if(light.lock()->getActive())
            {
                bufferData.push_back(light.lock()->getLightData());
            }
        }

        return bufferData;
    }

    return std::nullopt;
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::LightManager>("LightManager").constructor()(rttr::policy::ctor::as_std_shared_ptr);
}