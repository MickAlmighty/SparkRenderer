#ifndef LIGHT_MANAGER_H
#define LIGHT_MANAGER_H

#include <algorithm>
#include <memory>
#include <vector>

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

    std::vector<std::weak_ptr<LightProbe>> lightProbes;

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
    bool updateBuffer = false;

    template<typename T>
    void updateBufferIfNecessary(const std::vector<T>& bufferLightData, SSBO& ssbo);

    template<typename T>
    bool findDirtyLight(const std::vector<std::weak_ptr<T>>& lightContainer);

    template<typename T>
    bool findAndRemoveExpiredPointer(std::vector<std::weak_ptr<T>>& lightContainer);

    template<typename N, typename T>
    std::vector<N> getLightDataBuffer(std::vector<std::weak_ptr<T>>& lightContainer);
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};

template<typename T>
void LightManager::updateBufferIfNecessary(const std::vector<T>& bufferLightData, SSBO& ssbo)
{
    if(updateBuffer)
    {
        ssbo.updateData(bufferLightData);
        updateBuffer = false;
    }
}

template<typename T>
bool LightManager::findDirtyLight(const std::vector<std::weak_ptr<T>>& lightContainer)
{
    for(const std::weak_ptr<T>& light : lightContainer)
    {
        if(light.lock()->getDirty())
        {
            return true;
        }
    }
    return false;
}

template<typename T>
bool LightManager::findAndRemoveExpiredPointer(std::vector<std::weak_ptr<T>>& lightContainer)
{
    const auto containerIt =
        std::find_if(std::begin(lightContainer), std::end(lightContainer), [](const std::weak_ptr<T>& weakPtr) { return weakPtr.expired(); });

    if(containerIt != std::end(lightContainer))
    {
        lightContainer.erase(containerIt);
        return true;
    }
    return false;
}

template<typename N, typename T>
std::vector<N> LightManager::getLightDataBuffer(std::vector<std::weak_ptr<T>>& lightContainer)
{
    bool expiredPointer = false;
    do
    {
        expiredPointer = findAndRemoveExpiredPointer(lightContainer);
        if(expiredPointer)
        {
            updateBuffer = true;
        }
    }
    while(expiredPointer);

    if(updateBuffer)
    {
        std::vector<N> bufferData;
        for(const auto& light : lightContainer)
        {
            bufferData.push_back(light.lock()->getLightData());
        }

        return bufferData;
    }
    else
    {
        std::vector<N> bufferData;
        updateBuffer = findDirtyLight(lightContainer);
        if(updateBuffer)
        {
            for(const auto& light : lightContainer)
            {
                light.lock()->resetDirty();
                if(light.lock()->getActive())
                {
                    bufferData.push_back(light.lock()->getLightData());
                }
            }
        }
        return bufferData;
    }
}
}  // namespace spark

#endif