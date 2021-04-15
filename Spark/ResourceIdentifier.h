#pragma once

#include <filesystem>

#include "Resource.h"

namespace spark
{
namespace resourceManagement
{
    class ResourceIdentifier
    {
        public:
        ResourceIdentifier(const std::filesystem::path& fullResourcePath);
        ~ResourceIdentifier() = default;

        ResourceIdentifier(const ResourceIdentifier& identifier) = default;
        ResourceIdentifier(ResourceIdentifier&& identifier) noexcept;

        ResourceIdentifier operator=(const ResourceIdentifier& identifier) = delete;
        ResourceIdentifier operator=(ResourceIdentifier&& identifier) = delete;

        bool operator==(const ResourceIdentifier& identifier) const;
        bool operator<(const ResourceIdentifier& identifier) const;

        std::filesystem::path getFullPath() const;
        std::filesystem::path getDirectoryPath() const;
        std::filesystem::path getResourceName(bool withExtension = true) const;
        std::string getResourceExtension() const;
        std::string getResourceExtensionLowerCase() const;

        bool changeResourceDirectory(const std::filesystem::path& path);
        bool changeResourceName(const std::filesystem::path& name);
        std::shared_ptr<Resource> getResource();

        private:
        std::string extensionToLowerCase(const std::filesystem::path& path) const;

        std::filesystem::path resourcePath{};
        std::weak_ptr<Resource> resource;
    };
}  // namespace resourceManagement
}  // namespace spark
