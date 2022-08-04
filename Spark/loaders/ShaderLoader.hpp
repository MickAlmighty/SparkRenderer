#pragma once

#include <filesystem>
#include <memory>

#include "ILoader.hpp"

namespace spark::loaders
{
class ShaderLoader final : public ILoader
{
    public:
    std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                       const std::filesystem::path& resourceRelativePath) const override;

    bool isExtensionSupported(const std::string& ext) const override;
    std::vector<std::string> supportedExtensions() const override;

    ShaderLoader() = default;
    ~ShaderLoader() = default;
    ShaderLoader(const ShaderLoader&) = delete;
    ShaderLoader(const ShaderLoader&&) = delete;
    ShaderLoader& operator=(const ShaderLoader&) = delete;
    ShaderLoader& operator=(const ShaderLoader&&) = delete;
};
}  // namespace spark::loaders