#pragma once

#include "Component.h"

namespace spark
{
class Mesh;

class MeshPlane final : public Component
{
    public:
    MeshPlane();
    ~MeshPlane() override = default;
    MeshPlane(const MeshPlane&) = delete;
    MeshPlane(const MeshPlane&&) = delete;
    MeshPlane& operator=(const MeshPlane&) = delete;
    MeshPlane& operator=(const MeshPlane&&) = delete;

    void update() override;

    private:
    void setup();

    std::shared_ptr<Mesh> planeMesh{nullptr};

    RTTR_REGISTRATION_FRIEND
    RTTR_ENABLE(Component)
};

}  // namespace spark