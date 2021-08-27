#pragma once

#include <filesystem>

#include "Resource.h"

struct aiMesh;

namespace spark
{
namespace resourceManagement {
    class ResourceIdentifier;
}

class Mesh;

class ResourceLoader final
{
    public:
    static std::shared_ptr<resourceManagement::Resource> createHdrTexture(const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier);
    static std::shared_ptr<resourceManagement::Resource> createCompressedTexture(const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier);
    static std::shared_ptr<resourceManagement::Resource> createUncompressedTexture(const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier);
    static std::shared_ptr<resourceManagement::Resource> createModel(const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier);
    static std::shared_ptr<resourceManagement::Resource> createScene(const std::shared_ptr<resourceManagement::ResourceIdentifier>& resourceIdentifier);

    ResourceLoader(const ResourceLoader&) = delete;
    ResourceLoader(const ResourceLoader&&) = delete;
    ResourceLoader& operator=(const ResourceLoader&) = delete;
    ResourceLoader& operator=(const ResourceLoader&&) = delete;

    private:
    ResourceLoader() = default;
    ~ResourceLoader() = default;

    static std::shared_ptr<Mesh> loadMesh(aiMesh* assimpMesh, const std::filesystem::path& path);
};

}  // namespace spark