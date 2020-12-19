#include "LightManager.h"

#include "DirectionalLight.h"
#include "GameObject.h"
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
    const auto dirLightDataBufferOpt = getLightDataBuffer<DirectionalLightData, DirectionalLight>(directionalLights, dirLightSSBO);
    updateBufferIfNecessary(dirLightDataBufferOpt, dirLightSSBO);

    const auto pointLightDataBufferOpt = getLightDataBuffer<PointLightData, PointLight>(pointLights, pointLightSSBO);
    updateBufferIfNecessary(pointLightDataBufferOpt, pointLightSSBO);

    const auto spotLightDataBufferOpt = getLightDataBuffer<SpotLightData, SpotLight>(spotLights, spotLightSSBO);
    updateBufferIfNecessary(spotLightDataBufferOpt, spotLightSSBO);

    const auto lightProbesDataBufferOpt = getLightProbeDataBufer(lightProbes);
    updateBufferIfNecessary(lightProbesDataBufferOpt, lightProbeSSBO);
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
    const bool wasLightRemoved = removeExpiredLightPointers(lightProbeContainer);

    size_t lightProbesCounter{0};
    for(const auto& light : lightProbeContainer)
    {
        if(light.lock()->getGameObject()->isActive())
        {
            ++lightProbesCounter;
        }
    }

    const uint32_t lastNumberOfLightProbes = lightProbeSSBO.size / sizeof(LightProbeData);
    const bool isLightQuantityChanged = lastNumberOfLightProbes != lightProbesCounter;
    const bool isAtLeastOneLightDirty =
        std::any_of(lightProbeContainer.begin(), lightProbeContainer.end(), [](const std::weak_ptr<LightProbe>& light) {
            const auto lightProbe = light.lock();
            return lightProbe->getDirty() || !lightProbe->getGameObject()->isActive();
        });

    if(wasLightRemoved || isAtLeastOneLightDirty || isLightQuantityChanged)
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
            const auto lightProbe = light.lock();
            if(lightProbe->getActive() && lightProbe->getGameObject()->isActive())
            {
                bufferData.push_back(lightProbe->getLightData());
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