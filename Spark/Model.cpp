#include "Model.h"

#include "Logging.h"
#include "ResourceLibrary.h"
#include "Spark.h"
#include "Texture.h"
#include "Timer.h"

namespace spark::resources
{
Model::Model(const std::filesystem::path& path_, const std::vector<std::shared_ptr<Mesh>>& meshes_) : Resource(path_), meshes(meshes_) {}

std::vector<std::shared_ptr<Mesh>> Model::getMeshes() const
{
    return meshes;
}
}  // namespace spark::resources
