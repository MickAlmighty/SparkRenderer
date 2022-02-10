#pragma once

#include <filesystem>

namespace spark::resourceManagement
{
class Resource;
}

namespace spark::loaders
{
class GltfLoader final
{
    public:
    static std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                              const std::filesystem::path& resourceRelativePath);

    static bool isExtensionSupported(const std::string& ext);
    static std::vector<std::string> supportedExtensions();

    
    GltfLoader(const GltfLoader&) = delete;
    GltfLoader(const GltfLoader&&) = delete;
    GltfLoader& operator=(const GltfLoader&) = delete;
    GltfLoader& operator=(const GltfLoader&&) = delete;

    private:
    GltfLoader() = default;
    ~GltfLoader() = default;
};
}  // namespace spark::loaders
