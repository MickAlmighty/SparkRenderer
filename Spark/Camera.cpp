#include "Camera.h"
#include "JsonSerializer.h"
#include "Clock.h"
#include "HID.h"
#include "Spark.h"

namespace spark {

	Camera::Camera(glm::vec3 position, float yaw, float pitch) : Component("Camera"),
	Position(position), Front(glm::vec3(0.0f, 0.0f, -1.0f)), Yaw(yaw), Pitch(pitch) {
		updateCameraVectors();
	}

	Camera::Camera(float posX, float posY, float posZ, float yaw, float pitch) : Component("Camera"),
		Position(posX, posY, posZ), Front(glm::vec3(0.0f, 0.0f, -1.0f)), Yaw(yaw), Pitch(pitch) {
		updateCameraVectors();
	}

	glm::mat4 Camera::getViewMatrix() const {
		//return glm::lookAt(Position, Position + Front, Up);
		//cameraTarget = Position + Front;
		return glm::lookAt(Position, cameraTarget, WORLD_UP);
	}

	glm::mat4 Camera::getProjectionMatrix() const {
		return glm::perspectiveFov(glm::radians(fov), 1.0f * Spark::WIDTH, 1.0f * Spark::HEIGHT, zNear, zFar);
	}

	glm::vec3 Camera::getPosition() const { return Position; }

	void Camera::setProjectionMatrix(float fov, float nearPlane, float farPlane) {
		this->fov = fov;
		this->zNear = nearPlane;
		this->zFar = farPlane;
	}

	void Camera::setCameraTarget(glm::vec3 target) {
		if (cameraMode == CameraMode::ThirdPerson) {
			cameraTarget = target;
		}
	}

	glm::vec3 Camera::getCameraTarget() const { return cameraTarget; }
	glm::vec3 Camera::getFront() const { return Front; }
	glm::vec3 Camera::getUp() const { return Up; }
	glm::vec3 Camera::getRight() const { return Right; }
	float Camera::getYaw() const { return Yaw; }
	float Camera::getPitch() const { return Pitch; }
	float Camera::getMovementSpeed() const { return MovementSpeed; }
	float Camera::getMouseSensitivity() const { return MouseSensitivity; }
	float Camera::getZoom() const { return Zoom; }
	float Camera::getFov() const { return fov; }
	float Camera::getNearPlane() const { return zNear; }
	float Camera::getFarPlane() const { return zFar; }
	CameraMode Camera::getCameraMode() const { return cameraMode; }
	void Camera::setYaw(float yaw) { Yaw = yaw; }
	void Camera::setPitch(float pitch) { Pitch = pitch; }
	void Camera::setRotation(float yaw, float pitch) { Yaw = yaw; Pitch = pitch; }

	void Camera::processKeyboard() {
		if (cameraMode == CameraMode::FirstPerson) {
			processKeyboardFirstPerson();
		}
		if (cameraMode == CameraMode::ThirdPerson) {
			processKeyboardThirdPerson();
		}
	}

	void Camera::processKeyboardFirstPerson() {
		float velocity = MovementSpeed * static_cast<float>(Clock::getDeltaTime());
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
			finalDirection -= WORLD_UP;
		if (HID::isKeyPressed(GLFW_KEY_E))
			finalDirection += WORLD_UP;

		if (finalDirection != glm::vec3(0)) {
			finalDirection = glm::normalize(finalDirection);
			Position += finalDirection * velocity;
		}
	}

	void Camera::processKeyboardThirdPerson() {
		float velocity = MovementSpeed * static_cast<float>(Clock::getDeltaTime());
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

		if (finalDirection != glm::vec3(0)) {
			finalDirection = glm::normalize(finalDirection);
			Position += finalDirection * velocity;
		}
	}

	void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch) {
		static bool cameraRotation = false;
		if (HID::isKeyPressed(GLFW_KEY_SPACE)) {
			cameraRotation = !cameraRotation;
		}

		if (!cameraRotation)
			return;
		xoffset *= MouseSensitivity;
		yoffset *= MouseSensitivity;

		Yaw += xoffset;
		Pitch += yoffset;

		// Make sure that when pitch is out of bounds, screen doesn't get flipped
		if (constrainPitch) {
			if (Pitch > 89.0f)
				Pitch = 89.0f;
			if (Pitch < -89.0f)
				Pitch = -89.0f;
		}

		if (cameraMode == CameraMode::ThirdPerson) {
			processMouseMovementThirdPerson(xoffset, yoffset);
		}
	}

	void Camera::processMouseMovementThirdPerson(float xoffset, float yoffset) {
		float velocity = MovementSpeed * static_cast<float>(Clock::getDeltaTime());
		if (HID::isKeyPressed(GLFW_KEY_LEFT_SHIFT))
			velocity *= 1.5f;

		const float speed = 10;
		glm::vec3 finalDirection(0);
		finalDirection += Right * xoffset;
		finalDirection += Up * yoffset;

		if (finalDirection != glm::vec3(0)) {
			finalDirection = glm::normalize(finalDirection) * speed;
			Position += finalDirection * velocity;
		}
	}

	void Camera::processMouseScroll(float yoffset) {
		if (Zoom >= 1.0f && Zoom <= 45.0f)
			Zoom -= yoffset;
		if (Zoom <= 1.0f)
			Zoom = 1.0f;
		if (Zoom >= 45.0f)
			Zoom = 45.0f;
	}

	void Camera::updateCameraVectors() {
		if (cameraMode == CameraMode::FirstPerson) {
			updateCameraVectorsFirstPerson();
		}

		if (cameraMode == CameraMode::ThirdPerson) {
			updateCameraVectorsThirdPerson();
		}
	}

	void Camera::updateCameraVectorsFirstPerson() {
		// Calculate the new Front vector
		glm::vec3 front;
		front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
		front.y = sin(glm::radians(Pitch));
		front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
		Front = glm::normalize(front);
		//Front = glm::normalize(glm::vec3(front.x, 0, front.z));
		// Also re-calculate the Right and Up vector
		Right = glm::normalize(glm::cross(Front, WORLD_UP));
		// Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
		Up = glm::normalize(glm::cross(Right, Front));

		cameraTarget = Position + Front;
	}

	void Camera::updateCameraVectorsThirdPerson() {
		Front = glm::normalize(Position - cameraTarget);
		Right = glm::normalize(glm::cross(Front, WORLD_UP));
		Up = glm::normalize(glm::cross(Front, Right));
	}

	void Camera::update() {
		if (HID::isKeyPressed(GLFW_KEY_1))
			cameraMode = CameraMode::FirstPerson;
		if (HID::isKeyPressed(GLFW_KEY_2))
			cameraMode = CameraMode::ThirdPerson;
		// Update Front, Right and Up Vectors using the updated Euler angles
		updateCameraVectors();
	}

	void Camera::fixedUpdate() {}

	void Camera::drawGUI() {

	}

    COMPONENT_CONVERTER(Camera)
 }

RTTR_REGISTRATION{
    rttr::registration::enumeration<spark::CameraMode>("CameraMode")(
        rttr::value("FirstPerson", spark::CameraMode::FirstPerson),
        rttr::value("ThirdPerson", spark::CameraMode::ThirdPerson)
        );
	rttr::registration::class_<spark::Camera>("Camera")
    .constructor()(rttr::policy::ctor::as_std_shared_ptr)
	.property("cameraTarget", &spark::Camera::cameraTarget)
	.property("Position", &spark::Camera::Position)
	.property("Front", &spark::Camera::Front)
	.property("Up", &spark::Camera::Up)
	.property("Right", &spark::Camera::Right)
	.property("Yaw", &spark::Camera::Yaw)
	.property("Pitch", &spark::Camera::Pitch)
	.property("MovementSpeed", &spark::Camera::MovementSpeed)
	.property("MouseSensitivity", &spark::Camera::MouseSensitivity)
	.property("Zoom", &spark::Camera::Zoom)
	.property("cameraMode", &spark::Camera::cameraMode)
	.property("fov", &spark::Camera::fov)
	.property("zNear", &spark::Camera::zNear)
	.property("zFar", &spark::Camera::zFar);

    
    REGISTER_COMPONENT_CONVERTER(Camera)
}