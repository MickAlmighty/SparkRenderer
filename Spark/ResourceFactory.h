#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <functional>

namespace spark::resourceManagement
{
class Resource;
class ResourceIdentifier;

class ResourceFactory
{
    public:
    static std::shared_ptr<Resource> loadResource(const std::filesystem::path& resourcesRootPath, const std::filesystem::path& resourceRelativePath);
    static bool isExtensionSupported(const std::filesystem::path& filePath);

    static std::vector<std::string> supportedAnimationExtensions();
    static std::vector<std::string> supportedModelExtensions();
    static std::vector<std::string> supportedTextureExtensions();
    static std::vector<std::string> supportedShaderExtensions();
    static std::vector<std::string> supportedSceneExtensions();
    static std::vector<std::string> supportedExtensions();

    private:
    static std::string extensionToLowerCase(const std::filesystem::path& path);
};
}  // namespace spark::resourceManagement