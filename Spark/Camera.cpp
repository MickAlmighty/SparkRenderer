#include "Camera.h"

#include "JsonSerializer.h"
#include "Clock.h"
#include "HID.h"
#include "Spark.h"

namespace spark {

Camera::Camera(std::string&& newName) : Component(newName)
{
	MovementSpeed = SPEED;
	MouseSensitivity = SENSITIVITY;
	Zoom = ZOOM;
}

Camera::Camera(glm::vec3 position, glm::vec3 up, float yaw, float pitch) : Front(glm::vec3(0.0f, 0.0f, -1.0f)),
	MovementSpeed(SPEED),
	MouseSensitivity(SENSITIVITY), Zoom(ZOOM)
{
	Position = position;
	WorldUp = up;
	Yaw = yaw;
	Pitch = pitch;
	updateCameraVectors();
}

Camera::Camera(float posX, float posY, float posZ, float upX, float upY, float upZ, float yaw, float pitch) :
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
	//return glm::lookAt(Position, Position + Front, Up);
	//cameraTarget = Position + Front;
	return glm::lookAt(Position, cameraTarget, WorldUp);
}

glm::mat4 Camera::getProjectionMatrix() const
{
	return glm::perspectiveFov(glm::radians(fov), 1.0f * Spark::WIDTH, 1.0f * Spark::HEIGHT, zNear, zFar);
}

void Camera::setProjectionMatrix(float fov, float nearPlane, float farPlane)
{
	this->fov = fov;
	this->zNear = nearPlane;
	this->zFar = farPlane;
}

void Camera::setCameraTarget(glm::vec3 target)
{
	if (cameraMode == CameraMode::ThirdPerson)
	{
		cameraTarget = target;
	}
}

void Camera::ProcessKeyboard()
{
	if (cameraMode == CameraMode::FirstPerson)
	{
		processKeyboardFirstPerson();
	}
	if (cameraMode == CameraMode::ThirdPerson)
	{
		processKeyboardThirdPerson();
	}
}

void Camera::processKeyboardFirstPerson()
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

void Camera::processKeyboardThirdPerson()
{
	float velocity = MovementSpeed * Clock::getDeltaTime();
	if (HID::isKeyPressed(GLFW_KEY_LEFT_SHIFT))
		velocity *= 1.5f;

	glm::vec3 finalDirection(0);

	if (HID::isKeyPressed(GLFW_KEY_W))
		finalDirection -= Front;
	if (HID::isKeyPressed(GLFW_KEY_S))
		finalDirection += Front;
	if (HID::isKeyPressed(GLFW_KEY_A))
		finalDirection += Right;
	if (HID::isKeyPressed(GLFW_KEY_D))
		finalDirection -= Right;
	if (HID::isKeyPressed(GLFW_KEY_Q))
		finalDirection += Up;
	if (HID::isKeyPressed(GLFW_KEY_E))
		finalDirection -= Up;

	if (finalDirection != glm::vec3(0))
	{
		finalDirection = glm::normalize(finalDirection);
		Position += finalDirection * velocity;
	}
}

void Camera::ProcessMouseMovement(float xoffset, float yoffset, bool constrainPitch)
{
	static bool cameraRotation = false;
	if (HID::isKeyPressed(GLFW_KEY_SPACE))
	{
		cameraRotation = !cameraRotation;
	}

	if (cameraRotation == false)
		return;
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

	if (cameraMode == CameraMode::ThirdPerson)
	{
		processMouseMovementThirdPerson(xoffset, yoffset);
	}
}

void Camera::processMouseMovementThirdPerson(float xoffset, float yoffset)
{
	float velocity = MovementSpeed * Clock::getDeltaTime();
	if (HID::isKeyPressed(GLFW_KEY_LEFT_SHIFT))
		velocity *= 1.5f;

	const float speed = 10;
	glm::vec3 finalDirection(0);
	finalDirection += Right * xoffset;
	finalDirection += Up * yoffset;

	if (finalDirection != glm::vec3(0))
	{
		finalDirection = glm::normalize(finalDirection) * speed;
		Position += finalDirection * velocity;
	}
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
	if (cameraMode == CameraMode::FirstPerson)
	{
		updateCameraVectorsFirstPerson();
	}

	if (cameraMode == CameraMode::ThirdPerson)
	{
		updateCameraVectorsThirdPerson();
	}
}

void Camera::updateCameraVectorsFirstPerson()
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

	cameraTarget = Position + Front;
}

void Camera::updateCameraVectorsThirdPerson()
{
	Front = glm::normalize(Position - cameraTarget);
	Right = glm::normalize(glm::cross(Front, WorldUp));
	Up = glm::normalize(glm::cross(Front, Right));
}

SerializableType Camera::getSerializableType()
{
	return SerializableType::SCamera;
}

Json::Value Camera::serialize()
{
	Json::Value root;
	root["cameraTarget"] = JsonSerializer::serializeVec3(cameraTarget);
	root["Position"] = JsonSerializer::serializeVec3(Position);
	root["Front"] = JsonSerializer::serializeVec3(Front);
	root["Up"] = JsonSerializer::serializeVec3(Up);
	root["Right"] = JsonSerializer::serializeVec3(Right);
	root["Yaw"] = Yaw;
	root["Pitch"] = Pitch;
	root["fov"] = fov;
	root["zNear"] = zNear;
	root["zFar"] = zFar;
	root["cameraMode"] = static_cast<int>(cameraMode);

	return root;
}

void Camera::deserialize(Json::Value& root)
{
	name = root.get("name", "Camera").asString();
	cameraTarget = JsonSerializer::deserializeVec3(root["cameraTarget"]);
	Position = JsonSerializer::deserializeVec3(root["Position"]);
	Front = JsonSerializer::deserializeVec3(root["Front"]);
	Up = JsonSerializer::deserializeVec3(root["Up"]);
	Right = JsonSerializer::deserializeVec3(root["Right"]);

	Yaw = root["Yaw"].asFloat();
	Pitch = root["Pitch"].asFloat();
	fov = root.get("fov", 60).asFloat();
	zNear = root.get("zNear", 0.1f).asFloat();
	zFar = root.get("zFar", 100.0f).asFloat();

	cameraMode = static_cast<CameraMode>(root["cameraMode"].asInt());
}

void Camera::update()
{

	if (HID::isKeyPressed(GLFW_KEY_1))
		cameraMode = CameraMode::FirstPerson;
	if (HID::isKeyPressed(GLFW_KEY_2))
		cameraMode = CameraMode::ThirdPerson;
	// Update Front, Right and Up Vectors using the updated Euler angles
	updateCameraVectors();
}

void Camera::fixedUpdate()
{
}

void Camera::drawGUI()
{

}

}