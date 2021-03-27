#pragma once

#include <vector>

#include "Component.h"

namespace spark
{
class Mesh;

namespace resources
{
    class Model;
}

class ModelMesh final : public Component
{
    public:
    ModelMesh();
    ModelMesh(const ModelMesh&) = delete;
    ModelMesh(const ModelMesh&&) = delete;
    ModelMesh& operator=(const ModelMesh&) = delete;
    ModelMesh& operator=(const ModelMesh&&) = delete;

    void setModel(const std::shared_ptr<resources::Model>& model_);
    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;
    std::string getModelPath() const;
    void setModelPath(const std::string modelPath);

    private:
    std::string modelPath;
    std::shared_ptr<resources::Model> model{nullptr};
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component);
};

}  // namespace spark