#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Buffer.hpp"
#include "Component.h"

namespace spark
{
constexpr glm::vec3 WORLD_UP{ 0.0f, 1.0f, 0.0f };

class Camera : public Component
{
    public:
    Camera();
    explicit Camera(glm::vec3 position);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjection() const;
    glm::mat4 getProjectionReversedZ() const;

    const UniformBuffer& getUbo() const;

    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    glm::vec3 cameraTarget{ 0 };

    // Camera Attributes
    glm::vec3 position{0.0f};
    glm::vec3 front{ 0.0f, 0.0f, -1.0f };
    glm::vec3 up{ WORLD_UP };
    glm::vec3 right{};
    // Euler Angles
    float yaw{ -90.0f };
    float pitch{ 0.0f };
    // Camera options
    float movementSpeed{ 3.5f };
    float mouseSensitivity{ 0.1f };

    // perspective
    float fov = 60;
    float zNear = 0.1f;
    float zFar = 10000.0f;

    private:
    void processKeyboard();
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);

    glm::mat4 getProjectionReversedZInfiniteFarPlane() const;

    // Calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors();

    UniformBuffer cameraUbo{};

    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component)
};

}  // namespace spark