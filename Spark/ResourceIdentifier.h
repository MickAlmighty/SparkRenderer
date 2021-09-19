#pragma once

#include <filesystem>

#include "Resource.h"

namespace spark
{
namespace resourceManagement
{
    class ResourceIdentifier : public std::enable_shared_from_this<ResourceIdentifier>
    {
        public:
        ResourceIdentifier(std::filesystem::path pathToResources, std::filesystem::path relativePath);
        ~ResourceIdentifier() = default;

        ResourceIdentifier(const ResourceIdentifier& identifier) = default;
        ResourceIdentifier(ResourceIdentifier&& identifier) noexcept = default;

        ResourceIdentifier operator=(const ResourceIdentifier& identifier) = delete;
        ResourceIdentifier operator=(ResourceIdentifier&& identifier) = delete;

        bool operator==(const ResourceIdentifier& identifier) const;
        bool operator<(const ResourceIdentifier& identifier) const;

        std::filesystem::path getResourcesRootPath() const;
        std::filesystem::path getFullPath() const;
        std::filesystem::path getRelativePath() const;
        std::filesystem::path getResourceName(bool withExtension = true) const;
        std::string getResourceExtension() const;
        std::string getResourceExtensionLowerCase() const;

        std::shared_ptr<Resource> getResource();

        private:
        std::string extensionToLowerCase(const std::filesystem::path& path) const;

        std::filesystem::path pathToResources{};
        std::filesystem::path relativePathToResource{};
        std::weak_ptr<Resource> resource;
    };
}  // namespace resourceManagement
}  // namespace spark
