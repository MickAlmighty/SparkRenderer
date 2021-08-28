#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "Component.h"
#include "LightManager.h"
#include "LightStatus.hpp"
#include "Observable.hpp"


namespace spark::lights
{
struct SpotLightData final
{
    alignas(16) glm::vec3 position;
    float cutOff;
    glm::vec3 color;  // strength baked into color
    float outerCutOff;
    glm::vec3 direction;
    float maxDistance;
    glm::vec4 boundingSphere;  // for cone culling approximation
};

class SpotLight final : public Component, public Observable<LightStatus<SpotLight>>
{
    public:
    SpotLight();
    ~SpotLight() override;
    SpotLight(const SpotLight&) = delete;
    SpotLight(const SpotLight&&) = delete;
    SpotLight& operator=(const SpotLight&) = delete;
    SpotLight& operator=(const SpotLight&&) = delete;

    SpotLightData getLightData() const;
    glm::vec3 getPosition() const;
    glm::vec3 getDirection() const;
    glm::vec3 getColor() const;
    float getColorStrength() const;
    float getSoftCutOffRatio() const;
    float getOuterCutOff() const;
    float getMaxDistance() const;

    void setColor(glm::vec3 color_);
    void setColorStrength(float strength);
    void setDirection(glm::vec3 direction_);
    void setSoftCutOffRatio(float softCutOffRatio_);
    void setOuterCutOff(float outerCutOff_);
    void setMaxDistance(float maxDistance_);
    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    private:
    void onActive() override;
    void onInactive() override;
    void notifyAbout(LightCommand command);

    glm::vec3 color{1};
    float colorStrength{1};
    glm::vec3 direction{0.0f, -1.0f, 0.0f};
    float softCutOffRatio{0.2f};
    float outerCutOff{45.0f};
    float maxDistance{1.0f};
    std::shared_ptr<LightManager> lightManager{nullptr};

    RTTR_ENABLE(Component)
};
}  // namespace spark::lights