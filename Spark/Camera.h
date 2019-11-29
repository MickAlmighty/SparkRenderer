#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Component.h"

namespace spark {

// Default camera values
constexpr float YAW = -90.0f;
constexpr float PITCH = 0.0f;
constexpr float SPEED = 2.5f;
constexpr float SENSITIVITY = 0.1f;
constexpr float ZOOM = 45.0f;


enum class CameraMode
{
	FirstPerson = 0,
	ThirdPerson = 1
};

class Camera : public Component
{
public:
    Camera();
	Camera(glm::vec3 position, float yaw = YAW, float pitch = PITCH);
	Camera(float posX, float posY, float posZ, float yaw, float pitch);

	glm::mat4 getViewMatrix() const;
	glm::mat4 getProjectionMatrix() const;
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
	float getZoom() const;
	float getFov() const;
	float getNearPlane() const;
	float getFarPlane() const;
	CameraMode getCameraMode() const;
	void setYaw(float yaw);
	void setPitch(float pitch);
	void setRotation(float yaw, float pitch);
	void processKeyboard();
	void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);
	void processMouseScroll(float yoffset);

	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;

private:
	glm::vec3 cameraTarget{ 0 };

	// Camera Attributes
	glm::vec3 Position{};
	glm::vec3 Front{};
	glm::vec3 Up{};
	glm::vec3 Right{};
	const glm::vec3 WORLD_UP{ 0.0f, 1.0f, 0.0f };
	// Euler Angles
	float Yaw{ YAW };
	float Pitch{ PITCH };
	// Camera options
	float MovementSpeed{ SPEED };
	float MouseSensitivity{ SENSITIVITY };
	float Zoom{ ZOOM };

	CameraMode cameraMode = CameraMode::FirstPerson;
	//perspective
	float fov = 60;
	float zNear = 0.1f;
	float zFar = 100.0f;

	// Calculates the front vector from the Camera's (updated) Euler Angles
	void updateCameraVectors();
	void updateCameraVectorsFirstPerson();
	void updateCameraVectorsThirdPerson();
	void processKeyboardFirstPerson();
	void processKeyboardThirdPerson();
	void processMouseMovementThirdPerson(float xoffset, float yoffset);
	RTTR_REGISTRATION_FRIEND;
	RTTR_ENABLE(Component)
};

}
#endif