#ifndef LIGHT_MANAGER_H
#define LIGHT_MANAGER_H

#include <algorithm>
#include <memory>
#include <vector>
#include <optional>
#include <set>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <rttr/registration_friend>
#include <rttr/registration>

#include "Structs.h"

namespace spark
{
class DirectionalLight;
class PointLight;
class SpotLight;
class LightProbe;

class LightManager
{
    public:
    SSBO dirLightSSBO{}, pointLightSSBO{}, spotLightSSBO{}, lightProbeSSBO{};

    std::vector<std::weak_ptr<DirectionalLight>> directionalLights;
    std::vector<std::weak_ptr<PointLight>> pointLights;
    std::vector<std::weak_ptr<SpotLight>> spotLights;

    struct compareLightProbes
    {
        bool operator()(const std::weak_ptr<LightProbe>& l, const std::weak_ptr<LightProbe>& r) const;
    };

    std::multiset<std::weak_ptr<LightProbe>, compareLightProbes> lightProbes;

    void addLightProbe(const std::shared_ptr<LightProbe>& lightProbe);

    void addDirectionalLight(const std::shared_ptr<DirectionalLight>& directionalLight);
    void addPointLight(const std::shared_ptr<PointLight>& pointLight);
    void addSpotLight(const std::shared_ptr<SpotLight>& spotLight);
    void updateLightBuffers();

    LightManager();
    ~LightManager();
    LightManager(const LightManager& lightManager) = delete;
    LightManager(const LightManager&& lightManager) = delete;
    LightManager& operator=(const LightManager& lightManager) = delete;
    LightManager&& operator=(const LightManager&& lightManager) = delete;

    private:
    template<typename T>
    void updateBufferIfNecessary(const std::optional<std::vector<T>>& bufferLightDataOpt, SSBO& ssbo);

    template<typename T>
    bool removeExpiredLightPointers(std::vector<std::weak_ptr<T>>& lightContainer);

    template<typename N, typename T>
    std::optional<std::vector<N>> getLightDataBuffer(std::vector<std::weak_ptr<T>>& lightContainer, const SSBO& ssbo);

    bool removeExpiredLightPointers(std::multiset<std::weak_ptr<LightProbe>, compareLightProbes>& lightContainer);
    std::optional<std::vector<LightProbeData>> LightManager::getLightProbeDataBufer(
        std::multiset<std::weak_ptr<LightProbe>, compareLightProbes>& lightProbeContainer);

    template<typename T>
    bool checkChangeInQuantity(const SSBO& ssboSize, const uint32_t entitySize, const std::vector<std::weak_ptr<T>>& container);

    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};

template<typename T>
void LightManager::updateBufferIfNecessary(const std::optional<std::vector<T>>& bufferLightDataOpt, SSBO& ssbo)
{
    if(bufferLightDataOpt.has_value())
    {
        ssbo.updateData(bufferLightDataOpt.value());
    }
}

template<typename T>
inline bool LightManager::removeExpiredLightPointers(std::vector<std::weak_ptr<T>>& lightContainer)
{
    bool isBufferDirty = false;
    bool atLeastOnExpiredPointerRemoved = false;
    do
    {
        const auto containerIt =
            std::find_if(std::begin(lightContainer), std::end(lightContainer), [](const std::weak_ptr<T>& weakPtr) { return weakPtr.expired(); });

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

template<typename DataTypeOfSSBO, typename T>
std::optional<std::vector<DataTypeOfSSBO>> LightManager::getLightDataBuffer(std::vector<std::weak_ptr<T>>& lightContainer, const SSBO& ssbo)
{
    const bool wasLightRemoved = removeExpiredLightPointers(lightContainer);
    const bool isLightQuantityChanged = checkChangeInQuantity(ssbo, sizeof(DataTypeOfSSBO), lightContainer);
    const bool isAtLeastOneLightDirty =
        std::any_of(lightContainer.begin(), lightContainer.end(), [](const std::weak_ptr<T>& light) { return light.lock()->getDirty(); });

    if(wasLightRemoved || isAtLeastOneLightDirty || isLightQuantityChanged)
    {
        std::vector<DataTypeOfSSBO> bufferData;
        bufferData.reserve(lightContainer.size());

        for(const auto& light : lightContainer)
        {
            light.lock()->resetDirty();
            if(light.lock()->getActive() && light.lock()->getGameObject()->isActive())
            {
                bufferData.push_back(light.lock()->getLightData());
            }
        }

        return bufferData;
    }

    return std::nullopt;
}

template<typename T>
bool LightManager::checkChangeInQuantity(const SSBO& ssbo, const uint32_t entitySize, const std::vector<std::weak_ptr<T>>& container)
{
    const uint32_t lastNumberOfLightProbes = ssbo.size / entitySize;

    size_t counter{0};
    for(const auto& light : container)
    {
        if(light.lock()->getGameObject()->isActive())
        {
            ++counter;
        }
    }

    return lastNumberOfLightProbes != counter;
}

}  // namespace spark

#endif