#include "Camera.hpp"

#include "GUI/ImGui/imgui.h"

#include "CameraManager.hpp"
#include "utils/CommonUtils.h"
#include "GameObject.h"
#include "Scene.h"

namespace spark
{
Camera::~Camera()
{
    if(const auto cm = cameraManager.lock(); cm)
        cm->removeCamera(this);
}

void Camera::start()
{
    cameraManager = getGameObject()->getScene()->getCameraManager();
    const auto camera = std::static_pointer_cast<Camera>(shared_from_this());
    cameraManager.lock()->addCamera(camera);
}

void Camera::update()
{
    if(const auto gameObject = getGameObject(); gameObject)
    {
        const glm::mat3 worldRotation = glm::transpose(glm::inverse(gameObject->transform.world.getMatrix()));
        const auto camFront = worldRotation * glm::vec3{0.0f, 0.0f, -1.0f};
        front = glm::normalize(glm::normalize(camFront));
        right = glm::normalize(glm::cross(front, WORLD_UP));
        up = glm::normalize(glm::cross(right, front));

        position = gameObject->transform.world.getPosition();
        cameraTarget = position + front;

        utils::updateCameraUBO(cameraUbo, getProjectionReversedZ(), getViewMatrix(), position, zNear, zFar);
    }
}

bool Camera::isMainCamera() const
{
    return cameraManager.lock()->getMainCamera().get() == this;
}

void Camera::setAsMainCamera()
{
    cameraManager.lock()->setMainCamera(std::static_pointer_cast<Camera>(shared_from_this()));
}

void Camera::drawUIBody()
{
    bool isMainCam = isMainCamera();
    if(ImGui::Checkbox("MainCamera", &isMainCam))
    {
        if(isMainCam)
        {
            setAsMainCamera();
        }
        else
        {
            cameraManager.lock()->setMainCamera(nullptr);
        }
    }
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::Camera>("Camera").constructor()(rttr::policy::ctor::as_std_shared_ptr);
    //.property("isMainCamera", &spark::Camera::isMainCamera, &spark::Camera::isMain);
}