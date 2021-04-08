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
        ResourceIdentifier(const ResourceIdentifier&& identifier) noexcept;

        ResourceIdentifier operator=(const ResourceIdentifier& identifier) = delete;
        ResourceIdentifier operator=(const ResourceIdentifier&& identifier) = delete;

        bool operator==(const ResourceIdentifier& identifier) const;
        bool operator<(const ResourceIdentifier& identifier) const;

        std::filesystem::path getFullPath() const;
        std::filesystem::path getDirectoryPath() const;
        std::filesystem::path getResourceName(bool withExtension = true) const;
        std::filesystem::path getResourceExtension() const;

        bool changeResourceDirectory(const std::filesystem::path& path);
        bool changeResourceName(const std::filesystem::path& name);
        std::shared_ptr<Resource> getResource();

        private:
        std::filesystem::path resourcePath{};
        std::weak_ptr<Resource> resource;
    };
}  // namespace resourceManagement
}  // namespace spark
