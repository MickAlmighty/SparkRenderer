#pragma once

#include <filesystem>
#include <memory>

#include "ILoader.hpp"

namespace spark::loaders
{
class AnimationLoader final : public ILoader
{
    public:
    std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                       const std::filesystem::path& resourceRelativePath) const override;

    bool isExtensionSupported(const std::string& ext) const override;
    std::vector<std::string> supportedExtensions() const override;

    AnimationLoader() = default;
    ~AnimationLoader() = default;
    AnimationLoader(const AnimationLoader&) = delete;
    AnimationLoader(const AnimationLoader&&) = delete;
    AnimationLoader& operator=(const AnimationLoader&) = delete;
    AnimationLoader& operator=(const AnimationLoader&&) = delete;
};
}  // namespace spark::loaders