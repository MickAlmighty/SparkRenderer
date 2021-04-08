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
    static std::shared_ptr<Resource> createResource(const std::filesystem::path& filePath);
    static bool isExtensionSupported(const std::filesystem::path& filePath);

    private:
    static std::map<std::filesystem::path, std::function<std::shared_ptr<Resource>(const std::filesystem::path& path)>> resourceCreationFunctions;
};
}