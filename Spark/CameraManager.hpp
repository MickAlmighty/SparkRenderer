#pragma once
#include <memory>
#include <vector>

namespace spark
{
class Camera;
class CameraManager
{
    public:
    void addCamera(const std::shared_ptr<Camera>& camera);
    void removeCamera(Camera* camera);
    void setMainCamera(const std::shared_ptr<Camera>& camera);
    std::shared_ptr<Camera> getMainCamera() const;

    private:
    std::weak_ptr<Camera> mainCamera{};

    std::vector<std::weak_ptr<Camera>> cameras{};
};
}  // namespace spark
