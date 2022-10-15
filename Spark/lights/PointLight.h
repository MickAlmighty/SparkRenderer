#pragma once

#include <glm/glm.hpp>

#include "Component.h"
#include "LightStatus.hpp"
#include "Observable.hpp"

namespace spark
{
class Mesh;
}

namespace spark::lights
{
class LightManager;

struct PointLightData final
{
    alignas(16) glm::vec4 positionAndRadius;
    alignas(16) glm::vec3 color;  // strength baked into color
};

class PointLight final : public Component, public Observable<LightStatus<PointLight>>
{
    public:
    PointLight();
    ~PointLight() override;
    PointLight(const PointLight&) = delete;
    PointLight(const PointLight&&) = delete;
    PointLight& operator=(const PointLight&) = delete;
    PointLight& operator=(const PointLight&&) = delete;

    PointLightData getLightData() const;
    glm::vec3 getPosition() const;
    glm::vec3 getColor() const;
    float getColorStrength() const;
    float getRadius() const;
    glm::mat4 getLightModel() const;
    void setRadius(float radius);
    void setColor(glm::vec3 color_);
    void setColorStrength(float strength);
    void setLightModel(glm::mat4 model);

    void update() override;
    void drawUIBody() override;
    void start() override;

    private:
    void onActive() override;
    void onInactive() override;
    void notifyAbout(LightCommand command);

    std::shared_ptr<Mesh> sphere{nullptr};

    glm::vec3 color{1};
    float radius{1.0f};
    float colorStrength{1};
    glm::mat4 lightModel{1};

    RTTR_REGISTRATION_FRIEND
    RTTR_ENABLE(Component)
};
}  // namespace spark::lights
