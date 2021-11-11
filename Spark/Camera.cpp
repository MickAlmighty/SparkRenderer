#include "Camera.h"

#include "CommonUtils.h"
#include "Clock.h"
#include "HID/HID.h"
#include "JsonSerializer.h"
#include "Spark.h"

namespace spark
{
Camera::Camera() : Component("Camera")
{
    updateCameraVectors();
}

Camera::Camera(glm::vec3 position) : Component("Camera"), position(position)
{
    updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix() const
{
    return glm::lookAt(position, cameraTarget, up);
}

glm::mat4 Camera::getProjection() const
{
    return glm::perspectiveFov(glm::radians(fov), Spark::get().WIDTH * 1.0f, Spark::get().HEIGHT * 1.0f, zNear, zFar);
}

glm::mat4 Camera::getProjectionReversedZInfiniteFarPlane() const
{
    return utils::getProjectionReversedZInfFar(Spark::get().WIDTH, Spark::get().HEIGHT, fov, zNear);
}

glm::mat4 Camera::getProjectionReversedZ() const
{
    return utils::getProjectionReversedZ(Spark::get().WIDTH, Spark::get().HEIGHT, fov, zNear, zFar);
}
const UniformBuffer& Camera::getUbo() const
{
    return cameraUbo;
}

void Camera::processKeyboard()
{
    if (HID::isKeyPressedOrDown(Key::MOUSE_RIGHT))
    {
        if (HID::mouse.getScrollStatus() == ScrollStatus::POSITIVE)
            movementSpeed *= 1.2f;
        else if (HID::mouse.getScrollStatus() == ScrollStatus::NEGATIVE)
            movementSpeed *= .8f;

        if (movementSpeed < 0.0f)
            movementSpeed = 0.0f;
    }
    else if (HID::getKeyState(Key::MOUSE_RIGHT) == State::NONE)
    {
        if (HID::mouse.getScrollStatus() == ScrollStatus::POSITIVE)
            position += front;
        else if (HID::mouse.getScrollStatus() == ScrollStatus::NEGATIVE)
            position -= front;
    }

    const float velocity = movementSpeed * static_cast<float>(Clock::getDeltaTime());

    const glm::vec3 f = glm::normalize(glm::vec3(front.x, 0, front.z));
    const glm::vec3 r = glm::normalize(glm::vec3(right.x, 0, right.z));

    glm::vec3 finalDirection(0);

    if (HID::isKeyPressedOrDown(Key::W))
        finalDirection += f;
    if (HID::isKeyPressedOrDown(Key::S))
        finalDirection -= f;
    if (HID::isKeyPressedOrDown(Key::A))
        finalDirection -= r;
    if (HID::isKeyPressedOrDown(Key::D))
        finalDirection += r;
    if (HID::isKeyPressedOrDown(Key::Q))
        finalDirection -= WORLD_UP;
    if (HID::isKeyPressedOrDown(Key::E))
        finalDirection += WORLD_UP;

    if (finalDirection != glm::vec3(0))
    {
        finalDirection = glm::normalize(finalDirection);
        position += finalDirection * velocity;
    }
}

void Camera::processMouseMovement(float xoffset, float yoffset, bool constrainPitch)
{
    static bool cameraRotation = false;

    if(HID::isKeyReleased(Key::MOUSE_RIGHT) && cameraRotation)
    {
        glfwSetInputMode(Spark::get().getRenderingContext().window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        cameraRotation = false;
    }
    else if(HID::isKeyPressedOrDown(Key::MOUSE_RIGHT) && !cameraRotation)
    {
        glfwSetInputMode(Spark::get().getRenderingContext().window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        cameraRotation = true;
    }

    if(!cameraRotation)
        return;
    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;

    yaw += xoffset;
    pitch += yoffset;

    // Make sure that when pitch is out of bounds, screen doesn't get flipped
    if(constrainPitch)
    {
        if(pitch > 89.0f)
            pitch = 89.0f;
        if(pitch < -89.0f)
            pitch = -89.0f;
    }
}

void Camera::updateCameraVectors()
{
    // Calculate the new front vector
    glm::vec3 f;
    f.x = cos(glm::radians(pitch)) * cos(glm::radians(yaw));
    f.y = sin(glm::radians(pitch));
    f.z = cos(glm::radians(pitch)) * sin(glm::radians(yaw));
    front = glm::normalize(f);
    // Also re-calculate the right and up vector
    right = glm::normalize(glm::cross(front, WORLD_UP));
    // Normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    up = glm::normalize(glm::cross(right, front));

    cameraTarget = position + front;
}

void Camera::update()
{
    processKeyboard();
    processMouseMovement(HID::mouse.direction.x, -HID::mouse.direction.y);

    updateCameraVectors();

    utils::updateCameraUBO(cameraUbo, getProjectionReversedZ(), getViewMatrix(), position, zNear, zFar);
}

void Camera::fixedUpdate() {}

void Camera::drawGUI() {}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::Camera>("Camera")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("cameraTarget", &spark::Camera::cameraTarget)
        .property("position", &spark::Camera::position)
        .property("front", &spark::Camera::front)
        .property("up", &spark::Camera::up)
        .property("right", &spark::Camera::right)
        .property("yaw", &spark::Camera::yaw)
        .property("pitch", &spark::Camera::pitch)
        .property("movementSpeed", &spark::Camera::movementSpeed)
        .property("mouseSensitivity", &spark::Camera::mouseSensitivity)
        .property("fov", &spark::Camera::fov)
        .property("zNear", &spark::Camera::zNear)
        .property("zFar", &spark::Camera::zFar);
}