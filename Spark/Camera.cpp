#include <Camera.h>
#include <Clock.h>
#include <HID.h>


Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch): Front(glm::vec3(0.0f, 0.0f, -1.0f)),
                                                                          MovementSpeed(SPEED),
                                                                          MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
	Position = position;
	WorldUp = up;
	Yaw = yaw;
	Pitch = pitch;
	updateCameraVectors();
}

Camera::Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch):
	Front(glm::vec3(0.0f, 0.0f, -1.0f)), MovementSpeed(SPEED), MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
	Position = glm::vec3(posX, posY, posZ);
	WorldUp = glm::vec3(upX, upY, upZ);
	Yaw = yaw;
	Pitch = pitch;
	updateCameraVectors();
}

glm::mat4 Camera::GetViewMatrix()
{
	return glm::lookAt(Position, Position + Front, Up);
}

void Camera::ProcessKeyboard()
{
	float velocity = MovementSpeed * Clock::getDeltaTime();
	if (HID::isKeyPressed(GLFW_KEY_LEFT_SHIFT))
		velocity *= 1.5f;
	glm::vec3 front = glm::normalize(glm::vec3(Front.x, 0, Front.z));
	glm::vec3 right = glm::normalize(glm::vec3(Right.x, 0, Right.z));
	
	glm::vec3 finalDirection(0);

	if (HID::isKeyPressed(GLFW_KEY_W))
		finalDirection += front;
	if (HID::isKeyPressed(GLFW_KEY_S))
		finalDirection -= front;
	if (HID::isKeyPressed(GLFW_KEY_A))
		finalDirection -= right;
	if (HID::isKeyPressed(GLFW_KEY_D))
		finalDirection += right;
	if (HID::isKeyPressed(GLFW_KEY_Q))
		finalDirection -= WorldUp;
	if (HID::isKeyPressed(GLFW_KEY_E))
		finalDirection += WorldUp;

	if (finalDirection != glm::vec3(0))
	{
		finalDirection = glm::normalize(finalDirection);
		Position += finalDirection * velocity;
	}
}

void Camera::ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch)
{
	xoffset *= MouseSensitivity;
	yoffset *= MouseSensitivity;

	Yaw += xoffset;
	Pitch += yoffset;

	// Make sure that when pitch is out of bounds, screen doesn't get flipped
	if (constrainPitch)
	{
		if (Pitch > 89.0f)
			Pitch = 89.0f;
		if (Pitch < -89.0f)
			Pitch = -89.0f;
	}

	// Update Front, Right and Up Vectors using the updated Euler angles
	updateCameraVectors();
}

void Camera::ProcessMouseScroll(float yoffset)
{
	if (Zoom >= 1.0f && Zoom <= 45.0f)
		Zoom -= yoffset;
	if (Zoom <= 1.0f)
		Zoom = 1.0f;
	if (Zoom >= 45.0f)
		Zoom = 45.0f;
}

void Camera::updateCameraVectors()
{
	// Calculate the new Front vector
	glm::vec3 front;
	front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
	front.y = sin(glm::radians(Pitch));
	front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
	Front = glm::normalize(front);
	//Front = glm::normalize(glm::vec3(front.x, 0, front.z));
	// Also re-calculate the Right and Up vector
	Right = glm::normalize(glm::cross(Front, WorldUp));
	// Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
	Up = glm::normalize(glm::cross(Right, Front));
}
