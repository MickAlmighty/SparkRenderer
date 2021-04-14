#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <functional>

namespace spark::resourceManagement
{
class Resource;

class ResourceFactory
{
    public:
    static std::shared_ptr<Resource> createResource(const std::filesystem::path& filePath);
    static bool isExtensionSupported(const std::filesystem::path& filePath);

    static std::vector<std::string> supportedModelExtensions();
    static std::vector<std::string> supportedTextureExtensions();
    static std::vector<std::string> supportedShaderExtensions();
    static std::vector<std::string> supportedSceneExtensions();
    static std::vector<std::string> supportedExtensions();

    private:
    static std::map<std::string, std::function<std::shared_ptr<Resource>(const std::filesystem::path& path)>> resourceCreationFunctions;
};
}