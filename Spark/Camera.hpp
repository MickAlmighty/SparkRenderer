#pragma once
#include "Component.h"
#include "ICamera.hpp"

namespace spark
{
class CameraManager;

class Camera : public ICamera, public Component
{
    public:
    ~Camera() override;

    void start() override;
    void update() override;

    bool isMainCamera() const;
    void setAsMainCamera();

    protected:
    void drawUIBody() override;

    private:
    std::weak_ptr<CameraManager> cameraManager{};
    bool isMain{false};

    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(ICamera, Component)
};
}  // namespace spark