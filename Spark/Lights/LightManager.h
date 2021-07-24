#pragma once

#include <algorithm>
#include <memory>
#include <vector>
#include <optional>

#include <rttr/registration_friend>
#include <rttr/registration>

#include "Buffer.hpp"
#include "IObserver.hpp"
#include "LightStatus.hpp"

namespace spark::lights
{
class DirectionalLight;
class PointLight;
class SpotLight;
class LightProbe;

class LightManager : public IObserver<LightStatus<DirectionalLight>>,
                     public IObserver<LightStatus<PointLight>>,
                     public IObserver<LightStatus<SpotLight>>,
                     public IObserver<LightStatus<LightProbe>>
{
    public:
    void updateLightBuffers();

    const SSBO& getDirLightSSBO() const;
    const SSBO& getPointLightSSBO() const;
    const SSBO& getSpotLightSSBO() const;
    const SSBO& getLightProbeSSBO() const;
    const std::vector<DirectionalLight*>& getDirLights() const;
    const std::vector<PointLight*>& getPointLights() const;
    const std::vector<SpotLight*>& getSpotLights() const;
    const std::vector<LightProbe*>& getLightProbes() const;

    LightManager() = default;
    ~LightManager() override = default;
    LightManager(const LightManager& lightManager) = delete;
    LightManager(const LightManager&& lightManager) = delete;
    LightManager& operator=(const LightManager& lightManager) = delete;
    LightManager&& operator=(const LightManager&& lightManager) = delete;

    private:
    bool areDirLightsDirty{false};
    bool arePointLightsDirty{false};
    bool areSpotLightsDirty{false};
    bool areLightProbesDirty{false};

    SSBO dirLightSSBO{}, pointLightSSBO{}, spotLightSSBO{}, lightProbeSSBO{};
    std::vector<DirectionalLight*> directionalLights;
    std::vector<PointLight*> pointLights;
    std::vector<SpotLight*> spotLights;
    std::vector<LightProbe*> lightProbes;

    void updateDirLightBuffer();
    void updatePointLightBuffer();
    void updateSpotLightBuffer();
    void updateLightProbeBuffer();

    void update(const LightStatus<DirectionalLight>* const dirLightStatus) override;
    void update(const LightStatus<PointLight>* const pointLightStatus) override;
    void update(const LightStatus<SpotLight>* const spotLightStatus) override;
    void update(const LightStatus<LightProbe>* const lightProbeStatus) override;

    template<typename LightData, typename LightType>
    std::vector<LightData> prepareLightDataBuffer(const std::vector<LightType*>& lights);

    template<typename Light>
    void processLightStatus(const LightStatus<Light>* const lightStatus, bool& areLightsDirty, std::vector<Light*>& lights);

    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};

template<typename LightData, typename LightType>
inline std::vector<LightData> LightManager::prepareLightDataBuffer(const std::vector<LightType*>& lights)
{
    std::vector<LightData> bufferData;
    bufferData.reserve(lights.size());

    for(const auto& light : lights)
    {
        bufferData.push_back(light->getLightData());
    }

    return bufferData;
}

template<typename Light>
void LightManager::processLightStatus(const LightStatus<Light>* const lightStatus, bool& areLightsDirty, std::vector<Light*>& lights)
{
    switch(lightStatus->command)
    {
        case LightCommand::add:
            if(std::none_of(lights.cbegin(), lights.cend(), [&lightStatus](const Light* lightPtr) { return lightPtr == lightStatus->light; }))
            {
                lights.push_back(lightStatus->light);
                areLightsDirty = true;
            }
            break;
        case LightCommand::update:
            areLightsDirty = true;
            break;
        case LightCommand::remove:
            const auto it = std::find(lights.begin(), lights.end(), lightStatus->light);
            if(it != lights.end())
            {
                lights.erase(it);
                areLightsDirty = true;
            }
            break;
    }
}
}  // namespace spark::lights