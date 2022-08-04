#pragma once

#include <filesystem>
#include <memory>
#include <vector>

namespace spark::resourceManagement
{
class Resource;
}

namespace spark::loaders
{
class ILoader
{
    public:
    virtual std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                               const std::filesystem::path& resourceRelativePath) const = 0;
    virtual bool isExtensionSupported(const std::string& ext) const = 0;
    virtual std::vector<std::string> supportedExtensions() const = 0;

    virtual ~ILoader() = default;
};
}  // namespace spark::loaders