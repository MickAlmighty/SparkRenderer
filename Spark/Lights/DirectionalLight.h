#pragma once

#include <glm/glm.hpp>

#include "Component.h"
#include "LightStatus.hpp"
#include "Observable.hpp"

namespace spark::lights
{
class LightManager;

struct DirectionalLightData final
{
    alignas(16) glm::vec3 direction;
    alignas(16) glm::vec3 color;  // strength baked into color
};

class DirectionalLight final : public Component, public Observable<LightStatus<DirectionalLight>>
{
    public:
    DirectionalLight() = default;
    ~DirectionalLight() override;
    DirectionalLight(const DirectionalLight&) = delete;
    DirectionalLight(const DirectionalLight&&) = delete;
    DirectionalLight& operator=(const DirectionalLight&) = delete;
    DirectionalLight& operator=(const DirectionalLight&&) = delete;

    void update() override;
    void start() override;
    void drawUIBody() override;

    DirectionalLightData getLightData() const;
    glm::vec3 getDirection() const;
    glm::vec3 getColor() const;
    float getColorStrength() const;
    void setDirection(glm::vec3 direction_);
    void setColor(glm::vec3 color_);
    void setColorStrength(float strength);

    bool areLightShaftsEnabled() const;
    void setLightShafts(bool state);

    static DirectionalLight* getDirLightForLightShafts();

    private:
    void onActive() override;
    void onInactive() override;
    void notifyAbout(LightCommand command);

    void activateLightShafts();
    void deactivateLightShafts();

    inline static DirectionalLight* dirLightForLightShafts{nullptr};

    glm::vec3 dirLightFront{0.0f, -1.0f, 0.0f};
    glm::vec3 direction{0.0f, -1.0f, 0.0f};
    glm::vec3 color{1};
    float colorStrength{1};

    bool lightShaftsActive{false};
    RTTR_REGISTRATION_FRIEND
    RTTR_ENABLE(Component)
};
}  // namespace spark::lights