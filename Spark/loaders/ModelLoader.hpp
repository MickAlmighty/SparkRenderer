#pragma once

#include <filesystem>
#include <map>
#include <vector>

#include "ILoader.hpp"

struct aiMesh;

namespace spark
{
enum class TextureTarget : unsigned char;
class Mesh;
}  // namespace spark

namespace spark::loaders
{
class ModelLoader final : public ILoader
{
    public:
    std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                       const std::filesystem::path& resourceRelativePath) const override;

    bool isExtensionSupported(const std::string& ext) const override;
    std::vector<std::string> supportedExtensions() const override;

    ModelLoader() = default;
    ~ModelLoader() = default;
    ModelLoader(const ModelLoader&) = delete;
    ModelLoader(const ModelLoader&&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&&) = delete;

    private:
    static std::shared_ptr<spark::Mesh> loadMesh(aiMesh* assimpMesh, std::vector<std::map<spark::TextureTarget, std::string>>& materials,
                                                 const std::filesystem::path& path);
};
}  // namespace spark::loaders