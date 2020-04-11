#pragma once

#include <glm/glm.hpp>

#include "Component.h"


namespace spark
{
struct PointLightData;
class Mesh;

class PointLight final : public Component
{
    public:
    PointLight();
    ~PointLight();
    PointLight(const PointLight&) = delete;
    PointLight(const PointLight&&) = delete;
    PointLight& operator=(const PointLight&) = delete;
    PointLight& operator=(const PointLight&&) = delete;

    PointLightData getLightData() const;
    bool getDirty() const;
    glm::vec3 getPosition() const;
    glm::vec3 getColor() const;
    float getColorStrength() const;
    float getRadius() const;
    glm::mat4 getLightModel() const;
    void resetDirty();
    void setRadius(float radius);
    void setColor(glm::vec3 color_);
    void setColorStrength(float strength);
    void setLightModel(glm::mat4 model);
    void setActive(bool active_) override;
    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    private:
    std::shared_ptr<Mesh> sphere{nullptr};
    bool dirty = true;
    bool addedToLightManager = false;

    glm::vec3 color{1};
    float radius{1.0f};
    float colorStrength{1};
    glm::mat4 lightModel{1};

    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component);
};
}  // namespace spark
