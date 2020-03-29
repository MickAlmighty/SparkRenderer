#ifndef MODEL_H
#define MODEL_H
#include "GPUResource.h"

#include "Mesh.h"
#include "Resource.h"

struct aiMesh;

namespace spark::resources
{
class Model : public resourceManagement::Resource, public resourceManagement::GPUResource
{
    public:
    Model(const resourceManagement::ResourceIdentifier& identifier);

    bool isResourceReady() override;
    bool load() override;
    bool unload() override;

    bool gpuLoad() override;
    bool gpuUnload() override;

    [[nodiscard]] std::vector<std::shared_ptr<Mesh>> getMeshes() const;

    private:
    std::vector<std::shared_ptr<Mesh>> meshes;

    std::vector<std::shared_ptr<Mesh>> loadModel(const std::filesystem::path& path);
    std::shared_ptr<Mesh> loadMesh(aiMesh* assimpMesh, const std::filesystem::path& modelPath);
};
}

#endif