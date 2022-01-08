#pragma once

#include "ICamera.hpp"

namespace spark
{
class EditorCamera : public ICamera
{
    public:
    EditorCamera();
    explicit EditorCamera(glm::vec3 position_);
    ~EditorCamera() override = default;

    void update();

    // Euler Angles
    float yaw{-90.0f};
    float pitch{0.0f};
    // Camera options
    float movementSpeed{3.5f};
    float mouseSensitivity{0.1f};

    private:
    void updateCameraVectors();

    void processKeyboard();
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);

    RTTR_REGISTRATION_FRIEND
    RTTR_ENABLE(ICamera)
};

}  // namespace spark