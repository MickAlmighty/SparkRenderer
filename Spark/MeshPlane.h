#pragma once

#include "Component.h"

namespace spark
{
namespace resources
{
    class Shader;
}

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
    std::shared_ptr<Mesh> planeMesh{nullptr};

    void setup();

    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component);
};

}  // namespace spark