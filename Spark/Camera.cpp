#include "Camera.h"

#include "CommonUtils.h"
#include "Clock.h"
#include "HID/HID.h"
#include "JsonSerializer.h"
#include "Spark.h"

namespace spark
{
Camera::Camera() : Component("Camera") {}

Camera::Camera(glm::vec3 position, float yaw, float pitch)
    : Component("Camera"), Position(position), Front(glm::vec3(0.0f, 0.0f, -1.0f)), Yaw(yaw), Pitch(pitch)
{
    updateCameraVectors();
}

Camera::Camera(float posX, float posY, float posZ, float yaw, float pitch)
    : Component("Camera"), Position(posX, posY, posZ), Front(glm::vec3(0.0f, 0.0f, -1.0f)), Yaw(yaw), Pitch(pitch)
{
    updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix() const
{
    // return glm::lookAt(Position, Position + Front, Up);
    // cameraTarget = Position + Front;
    return glm::lookAt(Position, cameraTarget, WORLD_UP);
}

glm::mat4 Camera::getProjection() const
{
    return glm::perspectiveFov(glm::radians(fov), Spark::WIDTH * 1.0f, Spark::HEIGHT * 1.0f, zNear, zFar);
}

glm::mat4 Camera::getProjectionReversedZInfiniteFarPlane() const
{
    return utils::getProjectionReversedZInfFar(Spark::WIDTH, Spark::HEIGHT, fov, zNear);
}

glm::mat4 Camera::getProjectionReversedZ() const
{
    return utils::getProjectionReversedZ(Spark::WIDTH, Spark::HEIGHT, fov, zNear, zFar);
}

glm::vec3 Camera::getPosition() const
{
    return Position;
}

void Camera::setProjectionMatrix(float fov, float nearPlane, float farPlane)
{
    this->fov = fov;
    this->zNear = nearPlane;
    this->zFar = farPlane;
}

void Camera::setCameraTarget(glm::vec3 target)
{
    if(cameraMode == CameraMode::ThirdPerson)
    {
        cameraTarget = target;
    }
}

glm::vec3 Camera::getCameraTarget() const
{
    return cameraTarget;
}

glm::vec3 Camera::getFront() const
{
    return Front;
}

glm::vec3 Camera::getUp() const
{
    return Up;
}

glm::vec3 Camera::getRight() const
{
    return Right;
}

float Camera::getYaw() const
{
    return Yaw;
}

float Camera::getPitch() const
{
    return Pitch;
}

float Camera::getMovementSpeed() const
{
    return MovementSpeed;
}

float Camera::getMouseSensitivity() const
{
    return MouseSensitivity;
}

float Camera::getZoom() const
{
    return Zoom;
}

float Camera::getFov() const
{
    return fov;
}

float Camera::getNearPlane() const
{
    return zNear;
}

float Camera::getFarPlane() const
{
    return zFar;
}

bool Camera::isDirty() const
{
    return dirty;
}

void Camera::cleanDirty()
{
    dirty = false;
}

CameraMode Camera::getCameraMode() const
{
    return cameraMode;
}

void Camera::setYaw(float yaw)
{
    Yaw = yaw;
}

void Camera::setPitch(float pitch)
{
    Pitch = pitch;
}

void Camera::setRotation(float yaw, float pitch)
{
    Yaw = yaw;
    Pitch = pitch;
}

void Camera::processKeyboard()
{
    if(cameraMode == CameraMode::FirstPerson)
    {
        processKeyboardFirstPerson();
    }
    if(cameraMode == CameraMode::ThirdPerson)
    {
        processKeyboardThirdPerson();
    }
}

void Camera::processKeyboardFirstPerson()
{
    if(HID::isKeyPressedOrDown(Key::MOUSE_RIGHT))
    {
        if(HID::mouse.getScrollStatus() == ScrollStatus::POSITIVE)
            MovementSpeed *= 1.2f;
        else if(HID::mouse.getScrollStatus() == ScrollStatus::NEGATIVE)
            MovementSpeed *= .8f;

        if(MovementSpeed < 0.0f)
            MovementSpeed = 0.0f;
    }
    else if(HID::getKeyState(Key::MOUSE_RIGHT) == State::NONE)
    {
        if(HID::mouse.getScrollStatus() == ScrollStatus::POSITIVE)
            Position += Front;
        else if(HID::mouse.getScrollStatus() == ScrollStatus::NEGATIVE)
            Position -= Front;
    }

    const float velocity = MovementSpeed * static_cast<float>(Clock::getDeltaTime());

    const glm::vec3 front = glm::normalize(glm::vec3(Front.x, 0, Front.z));
    const glm::vec3 right = glm::normalize(glm::vec3(Right.x, 0, Right.z));

    glm::vec3 finalDirection(0);

    if(HID::isKeyPressedOrDown(Key::W))
        finalDirection += front;
    if(HID::isKeyPressedOrDown(Key::S))
        finalDirection -= front;
    if(HID::isKeyPressedOrDown(Key::A))
        finalDirection -= right;
    if(HID::isKeyPressedOrDown(Key::D))
        finalDirection += right;
    if(HID::isKeyPressedOrDown(Key::Q))
        finalDirection -= WORLD_UP;
    if(HID::isKeyPressedOrDown(Key::E))
        finalDirection += WORLD_UP;

    if(finalDirection != glm::vec3(0))
    {
        finalDirection = glm::normalize(finalDirection);
        Position += finalDirection * velocity;
        dirty = true;
    }
}

void Camera::processKeyboardThirdPerson()
{
    float velocity = MovementSpeed * static_cast<float>(Clock::getDeltaTime());
    if(HID::isKeyPressedOrDown(Key::LEFT_SHIFT))
        velocity *= 3.5f;

    glm::vec3 finalDirection(0);

    if(HID::isKeyPressedOrDown(Key::A))
        finalDirection += Right;
    if(HID::isKeyPressedOrDown(Key::D))
        finalDirection -= Right;
    if(HID::isKeyPressedOrDown(Key::Q))
        finalDirection += Up;
    if(HID::isKeyPressedOrDown(Key::E))
        finalDirection -= Up;

    if(finalDirection != glm::vec3(0))
    {
        finalDirection = glm::normalize(finalDirection);
        glm::vec3 newPosition = Position + finalDirection * velocity;

        const float oldDistance = glm::length(cameraTarget - Position);
        const float newDistance = glm::length(cameraTarget - newPosition);
        const float diff = newDistance - oldDistance;

        glm::vec3 newDirection = glm::normalize(cameraTarget - newPosition);

        Position = newPosition + newDirection * diff;
        dirty = true;
    }

    finalDirection = glm::vec3(0.0f);
    if(HID::isKeyPressedOrDown(Key::W))
        finalDirection -= Front;
    if(HID::isKeyPressedOrDown(Key::S))
        finalDirection += Front;

    if(finalDirection != glm::vec3(0))
    {
        Position += finalDirection * velocity;
        dirty = true;
    }
}

void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch)
{
    static bool cameraRotation = false;

    if(HID::isKeyReleased(Key::MOUSE_RIGHT) && cameraRotation)
    {
        glfwSetInputMode(Spark::window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        cameraRotation = false;
    }
    else if(HID::isKeyPressedOrDown(Key::MOUSE_RIGHT) && !cameraRotation)
    {
        glfwSetInputMode(Spark::window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        cameraRotation = true;
    }

    if(!cameraRotation)
        return;
    xoffset *= MouseSensitivity;
    yoffset *= MouseSensitivity;

    if(xoffset != 0.0f || yoffset != 0.0f)
    {
        dirty = true;
    }

    Yaw += xoffset;
    Pitch += yoffset;

    // Make sure that when pitch is out of bounds, screen doesn't get flipped
    if(constrainPitch)
    {
        if(Pitch > 89.0f)
            Pitch = 89.0f;
        if(Pitch < -89.0f)
            Pitch = -89.0f;
    }

    if(cameraMode == CameraMode::ThirdPerson)
    {
        processMouseMovementThirdPerson(xoffset, yoffset);
    }
}

void Camera::processMouseMovementThirdPerson(float xoffset, float yoffset)
{
    float velocity = MovementSpeed * static_cast<float>(Clock::getDeltaTime());
    if(HID::isKeyPressedOrDown(Key::LEFT_SHIFT))
        velocity *= 1.5f;

    const float speed = 10;
    glm::vec3 finalDirection(0);
    finalDirection += Right * xoffset;
    finalDirection += Up * yoffset;

    if(finalDirection != glm::vec3(0))
    {
        finalDirection = glm::normalize(finalDirection);
        const float oldDistance = glm::length(cameraTarget - Position);

        const glm::vec3 newPosition = Position + finalDirection * velocity * oldDistance;

        const float newDistance = glm::length(cameraTarget - newPosition);
        const float diff = newDistance - oldDistance;

        const glm::vec3 newDirection = glm::normalize(cameraTarget - newPosition);

        Position = newPosition + newDirection * diff;
        dirty = true;
    }
}

void Camera::processMouseScroll(float yoffset)
{
    if(Zoom >= 1.0f && Zoom <= 45.0f)
        Zoom -= yoffset;
    if(Zoom <= 1.0f)
        Zoom = 1.0f;
    if(Zoom >= 45.0f)
        Zoom = 45.0f;
}

void Camera::updateCameraVectors()
{
    if(cameraMode == CameraMode::FirstPerson)
    {
        updateCameraVectorsFirstPerson();
    }

    if(cameraMode == CameraMode::ThirdPerson)
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
    // Front = glm::normalize(glm::vec3(front.x, 0, front.z));
    // Also re-calculate the Right and Up vector
    Right = glm::normalize(glm::cross(Front, WORLD_UP));
    // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    Up = glm::normalize(glm::cross(Right, Front));

    cameraTarget = Position + Front;
}

void Camera::updateCameraVectorsThirdPerson()
{
    Front = glm::normalize(Position - cameraTarget);
    Right = glm::normalize(glm::cross(Front, WORLD_UP));
    Up = glm::normalize(glm::cross(Front, Right));
}

void Camera::update()
{
    if(HID::isKeyPressed(Key::NUM_1))
        cameraMode = CameraMode::FirstPerson;
    if(HID::isKeyPressed(Key::NUM_2))
        cameraMode = CameraMode::ThirdPerson;
    // Update Front, Right and Up Vectors using the updated Euler angles
    updateCameraVectors();
}

void Camera::fixedUpdate() {}

void Camera::drawGUI() {}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::enumeration<spark::CameraMode>("CameraMode")(rttr::value("FirstPerson", spark::CameraMode::FirstPerson),
                                                                     rttr::value("ThirdPerson", spark::CameraMode::ThirdPerson));
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
}