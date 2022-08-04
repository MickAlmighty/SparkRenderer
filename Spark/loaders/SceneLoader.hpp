#pragma once

#include <filesystem>

#include "ILoader.hpp"

namespace spark::loaders
{
class SceneLoader final : public ILoader
{
    public:
    std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                       const std::filesystem::path& resourceRelativePath) const override;

    bool isExtensionSupported(const std::string& ext) const override;
    std::vector<std::string> supportedExtensions() const override;

    SceneLoader() = default;
    ~SceneLoader() = default;
    SceneLoader(const SceneLoader&) = delete;
    SceneLoader(const SceneLoader&&) = delete;
    SceneLoader& operator=(const SceneLoader&) = delete;
    SceneLoader& operator=(const SceneLoader&&) = delete;
};
}  // namespace spark::loaders