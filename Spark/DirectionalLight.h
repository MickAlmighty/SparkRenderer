#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include "Component.h"

#include <glm/glm.hpp>

namespace spark
{
struct DirectionalLightData;

class DirectionalLight final : public Component
{
    public:
    DirectionalLight();
    ~DirectionalLight() = default;
    DirectionalLight(const DirectionalLight&) = delete;
    DirectionalLight(const DirectionalLight&&) = delete;
    DirectionalLight& operator=(const DirectionalLight&) = delete;
    DirectionalLight& operator=(const DirectionalLight&&) = delete;

    DirectionalLightData getLightData() const;
    bool getDirty() const;
    glm::vec3 getDirection() const;
    glm::vec3 getColor() const;
    float getColorStrength() const;
    void resetDirty();
    void setDirection(glm::vec3 direction_);
    void setColor(glm::vec3 color_);
    void setColorStrength(float strength);
    void setActive(bool active_) override;
    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    private:
    // TODO: a lot of methods and fields in Light classes are copied.
    // This might and should be changed to a common abstract base class
    bool dirty = true;
    bool addedToLightManager = false;
    glm::vec3 direction{0.0f, -1.0f, 0.0f};
    glm::vec3 color{1};
    float colorStrength{1};
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component);
};
}  // namespace spark

#endif