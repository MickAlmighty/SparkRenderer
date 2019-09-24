#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <Component.h>

// Default camera values
const float YAW = -90.0f;
const float PITCH = 0.0f;
const float SPEED = 2.5f;
const float SENSITIVITY = 0.1f;
const float ZOOM = 45.0f;


enum class CameraMode
{
	FirstPerson,
	ThirdPerson
};

class Camera : public Component
{
private:
	glm::vec3 cameraTarget{ 0 };
	// Calculates the front vector from the Camera's (updated) Euler Angles
	void updateCameraVectors();
	void updateCameraVectorsFirstPerson();
	void updateCameraVectorsThirdPerson();
public:
	// Camera Attributes
	glm::vec3 Position{};
	glm::vec3 Front{};
	glm::vec3 Up{};
	glm::vec3 Right{};
	glm::vec3 WorldUp{0.0f, 1.0f, 0.0f};
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

	Camera(std::string&& newName = "Camera");
	Camera(glm::vec3 position, glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f),
	       float yaw = YAW, float pitch = PITCH);

	Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch);


	glm::mat4 GetViewMatrix();
	glm::mat4 getProjectionMatrix() const;
	void setProjectionMatrix(float fov, float nearPlane, float farPlane);
	void setCameraTarget(glm::vec3 target);
	void ProcessKeyboard();
	void processKeyboardFirstPerson();
	void processKeyboardThirdPerson();
	
	void ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch = true);
	void processMouseMovementThirdPerson(float xoffset, float yoffset);

	void ProcessMouseScroll(float yoffset);

	SerializableType getSerializableType() override;
	Json::Value serialize() override;
	void deserialize(Json::Value& root) override;
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;
};

#endif