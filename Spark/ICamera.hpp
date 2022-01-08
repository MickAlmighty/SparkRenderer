#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <rttr/registration>
#include <rttr/registration_friend.h>

#include "Buffer.hpp"

namespace spark
{
constexpr glm::vec3 WORLD_UP{0.0f, 1.0f, 0.0f};

class ICamera
{
    public:
    virtual ~ICamera() = default;

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjection() const;
    glm::mat4 getProjectionReversedZ() const;

    const UniformBuffer& getUbo() const;

    glm::vec3 position{0.0f};
    glm::vec3 front{0.0f, 0.0f, -1.0f};

    // perspective
    float fov = 60;
    float zNear = 0.1f;
    float zFar = 1000.0f;

    protected:
    glm::vec3 cameraTarget{0};

    glm::vec3 up{WORLD_UP};
    glm::vec3 right{1.0f, 0.0f, 0.0f};

    UniformBuffer cameraUbo{};

    private:
    glm::mat4 getProjectionReversedZInfiniteFarPlane() const;

    RTTR_REGISTRATION_FRIEND
    RTTR_ENABLE()
};
}  // namespace spark