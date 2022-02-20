#pragma once

#include <filesystem>
#include <map>
#include <vector>

struct aiMesh;

namespace spark
{
enum class TextureTarget : unsigned char;
class Mesh;

namespace resourceManagement
{
    class Resource;
}
}  // namespace spark

namespace spark::resourceManagement
{
class Resource;
}

namespace spark::loaders
{
class ModelLoader final
{
    public:
    static std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                              const std::filesystem::path& resourceRelativePath);

    static bool isExtensionSupported(const std::string& ext);
    static std::vector<std::string> supportedExtensions();

    ModelLoader(const ModelLoader&) = delete;
    ModelLoader(const ModelLoader&&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&&) = delete;

    private:
    ModelLoader() = default;
    ~ModelLoader() = default;

    static std::shared_ptr<spark::Mesh> loadMesh(aiMesh* assimpMesh, std::vector<std::map<spark::TextureTarget, std::string>>& materials,
                                                 const std::filesystem::path& path);
};
}  // namespace spark::loaders