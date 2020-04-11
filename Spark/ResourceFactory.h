#ifndef RESOURCE_FACTORY_H
#define RESOURCE_FACTORY_H

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <functional>

namespace spark::resourceManagement
{
class Resource;
class ResourceIdentifier;

class ResourceFactory
{
    public:
    static std::optional<std::shared_ptr<Resource>> createResource(const std::filesystem::path& filePath);

    private:
    static inline std::set<std::filesystem::path> supportedExtensions = {
        ".obj", ".dds", ".ktx", ".DDS", ".KTX", ".png", ".jpg", ".tga", "glsl"
    };

    static std::map<std::filesystem::path, std::function<std::shared_ptr<Resource>(const ResourceIdentifier & id)>> resourceCreationFunctions;
};
}
#endif