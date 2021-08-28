#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Buffer.hpp"
#include "Component.h"

namespace spark
{
enum class CameraMode
{
    FirstPerson = 0,
    ThirdPerson = 1
};

class Camera : public Component
{
    public:
    Camera();
    Camera(glm::vec3 position);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjection() const;
    glm::mat4 getProjectionReversedZInfiniteFarPlane() const;
    glm::mat4 getProjectionReversedZ() const;
    glm::vec3 getPosition() const;
    void setProjectionMatrix(float fov, float nearPlane, float farPlane);
    void setCameraTarget(glm::vec3 target);
    glm::vec3 getCameraTarget() const;
    glm::vec3 getFront() const;
    glm::vec3 getUp() const;
    glm::vec3 getRight() const;
    float getYaw() const;
    float getPitch() const;
    float getMovementSpeed() const;
    float getMouseSensitivity() const;
    float getFov() const;
    float getNearPlane() const;
    float getFarPlane() const;
    const UniformBuffer& getUbo();
    bool isDirty() const;
    void cleanDirty();
    CameraMode getCameraMode() const;
    void setYaw(float yaw);
    void setPitch(float pitch);
    void setRotation(float yaw, float pitch);

    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    private:
    void processKeyboard();
    void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);

    // Calculates the front vector from the Camera's (updated) Euler Angles
    void updateCameraVectors();
    void updateCameraVectorsFirstPerson();
    void updateCameraVectorsThirdPerson();
    void processKeyboardFirstPerson();
    void processKeyboardThirdPerson();
    void processMouseMovementThirdPerson(float xoffset, float yoffset);

    glm::vec3 cameraTarget{0};

    // Camera Attributes
    glm::vec3 Position{};
    glm::vec3 Front{};
    glm::vec3 Up{};
    glm::vec3 Right{};
    const glm::vec3 WORLD_UP{0.0f, 1.0f, 0.0f};
    // Euler Angles
    float Yaw{-90.0f};
    float Pitch{0.0f};
    // Camera options
    float MovementSpeed{3.5f};
    float MouseSensitivity{0.1f};
    bool dirty = true;

    CameraMode cameraMode = CameraMode::FirstPerson;
    // perspective
    float fov = 60;
    float zNear = 0.1f;
    float zFar = 10000.0f;

    UniformBuffer cameraUbo{};

    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component)
};

}  // namespace spark