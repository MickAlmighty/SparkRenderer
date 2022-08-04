#pragma once

#include <filesystem>

#include "ILoader.hpp"

namespace spark::loaders
{
class GltfLoader final : public ILoader
{
    public:
    std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                       const std::filesystem::path& resourceRelativePath) const override;

    bool isExtensionSupported(const std::string& ext) const override;
    std::vector<std::string> supportedExtensions() const override;

    GltfLoader() = default;
    ~GltfLoader() = default;
    GltfLoader(const GltfLoader&) = delete;
    GltfLoader(const GltfLoader&&) = delete;
    GltfLoader& operator=(const GltfLoader&) = delete;
    GltfLoader& operator=(const GltfLoader&&) = delete;
};
}  // namespace spark::loaders
