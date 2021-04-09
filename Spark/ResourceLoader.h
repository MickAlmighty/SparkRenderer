#pragma once

#include <filesystem>
#include <optional>

#include "Resource.h"

struct aiMesh;

namespace spark
{
struct PbrCubemapTexture;
class Mesh;

class ResourceLoader final
{
    public:
    static std::shared_ptr<resourceManagement::Resource> createHdrTexture(const std::filesystem::path& path);
    static std::shared_ptr<resourceManagement::Resource> createCompressedTexture(const std::filesystem::path& path);
    static std::shared_ptr<resourceManagement::Resource> createUncompressedTexture(const std::filesystem::path& path);
    static std::shared_ptr<resourceManagement::Resource> createModel(const std::filesystem::path& path);

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