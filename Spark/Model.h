#pragma once

#include "Mesh.h"
#include "Resource.h"



namespace spark::resources
{
class Model : public resourceManagement::Resource
{
    public:
    Model(const std::filesystem::path& path_, const std::vector<std::shared_ptr<Mesh>>& meshes_);

    [[nodiscard]] std::vector<std::shared_ptr<Mesh>> getMeshes() const;

    private:
    std::vector<std::shared_ptr<Mesh>> meshes;
};
}