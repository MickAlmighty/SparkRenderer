#pragma once

#include <filesystem>

namespace spark::resourceManagement
{
class Resource;
}

namespace spark::loaders
{
class SceneLoader final
{
    public:
    static std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                              const std::filesystem::path& resourceRelativePath);

    static bool isExtensionSupported(const std::string& ext);
    static std::vector<std::string> supportedExtensions();

    
    SceneLoader(const SceneLoader&) = delete;
    SceneLoader(const SceneLoader&&) = delete;
    SceneLoader& operator=(const SceneLoader&) = delete;
    SceneLoader& operator=(const SceneLoader&&) = delete;

    private:
    SceneLoader() = default;
    ~SceneLoader() = default;
};
}  // namespace spark::loaders