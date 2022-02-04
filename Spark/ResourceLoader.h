#pragma once

#include <filesystem>
#include <map>

#include "Resource.h"

struct aiMesh;

namespace spark
{
enum class TextureTarget : unsigned char;

namespace resourceManagement
{
    class ResourceIdentifier;
}

class Mesh;

class ResourceLoader final
{
    public:
    static std::shared_ptr<resourceManagement::Resource> createHdrTexture(const std::filesystem::path& resourcesRootPath,
                                                                          const std::filesystem::path& resourceRelativePath);
    static std::shared_ptr<resourceManagement::Resource> createCompressedTexture(const std::filesystem::path& resourcesRootPath,
                                                                                 const std::filesystem::path& resourceRelativePath);
    static std::shared_ptr<resourceManagement::Resource> createUncompressedTexture(const std::filesystem::path& resourcesRootPath,
                                                                                   const std::filesystem::path& resourceRelativePath);
    static std::shared_ptr<resourceManagement::Resource> createModel(const std::filesystem::path& resourcesRootPath,
                                                                     const std::filesystem::path& resourceRelativePath);
    static std::shared_ptr<resourceManagement::Resource> createShader(const std::filesystem::path& resourcesRootPath,
                                                                      const std::filesystem::path& resourceRelativePath);
    static std::shared_ptr<resourceManagement::Resource> createScene(const std::filesystem::path& resourcesRootPath,
                                                                     const std::filesystem::path& resourceRelativePath);
    static std::shared_ptr<resourceManagement::Resource> createAnimation(const std::filesystem::path& resourcesRootPath,
                                                                         const std::filesystem::path& resourceRelativePath);

    ResourceLoader(const ResourceLoader&) = delete;
    ResourceLoader(const ResourceLoader&&) = delete;
    ResourceLoader& operator=(const ResourceLoader&) = delete;
    ResourceLoader& operator=(const ResourceLoader&&) = delete;

    private:
    ResourceLoader() = default;
    ~ResourceLoader() = default;

    static std::shared_ptr<Mesh> loadMesh(aiMesh* assimpMesh, std::vector<std::map<TextureTarget, std::string>>& materials,
                                          const std::filesystem::path& path);
};

}  // namespace spark