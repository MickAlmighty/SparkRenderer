#include "CameraManager.hpp"

namespace spark
{
void CameraManager::addCamera(const std::shared_ptr<Camera>& camera)
{
    if(camera != nullptr)
    {
        if(cameras.empty())
        {
            mainCamera = camera;
        }
        cameras.push_back(camera);
    }
}

void CameraManager::removeCamera(Camera* camera)
{
    const auto mainCameraSharedPtr = mainCamera.lock();
    for(auto it = cameras.begin(); it != cameras.end();)
    {
        const auto cam = it->lock();
        if(!cam)
        {
            it = cameras.erase(it);
            continue;
        }

        if(cam.get() == camera)
        {
            cameras.erase(it);
            if(mainCameraSharedPtr && mainCameraSharedPtr.get() == camera)
            {
                mainCamera.reset();
            }

            break;
        }

        ++it;
    }
}

void CameraManager::setMainCamera(const std::shared_ptr<Camera>& camera)
{
    mainCamera = camera;
}

std::shared_ptr<Camera> CameraManager::getMainCamera() const
{
    return mainCamera.lock();
}
}  // namespace spark
