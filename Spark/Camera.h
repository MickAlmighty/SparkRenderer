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


enum class CameraMode : uint16_t
{
	FirstPerson,
	ThirdPerson
};

class Camera : public Component
{
public:
	Camera(std::string&& newName = "Camera");
	Camera(glm::vec3 position, glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH);
	Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);

	glm::mat4 getViewMatrix() const;
	glm::mat4 getProjectionMatrix() const;
	glm::vec3 getPosition() const;
	glm::vec3 getFront() const;
	float getFarPlane() const;
	float getNearPlane() const;
	void setProjectionMatrix(float fov, float nearPlane, float farPlane);
	void setCameraTarget(glm::vec3 target);
	void processKeyboard();
	void processMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);
	void processMouseScroll(float yoffset);

	SerializableType getSerializableType() override;
	Json::Value serialize() override;
	void deserialize(Json::Value& root) override;
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
	glm::vec3 WorldUp{ 0.0f, 1.0f, 0.0f };
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
};

}
#endif