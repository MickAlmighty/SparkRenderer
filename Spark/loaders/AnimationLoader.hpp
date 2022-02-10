#pragma once

#include <filesystem>
#include <memory>

namespace spark::resourceManagement
{
class Resource;
}

namespace spark::loaders
{
class AnimationLoader final
{
    public:
    static std::shared_ptr<resourceManagement::Resource> load(const std::filesystem::path& resourcesRootPath,
                                                              const std::filesystem::path& resourceRelativePath);

    static bool isExtensionSupported(const std::string& ext);
    static std::vector<std::string> supportedExtensions();

    AnimationLoader(const AnimationLoader&) = delete;
    AnimationLoader(const AnimationLoader&&) = delete;
    AnimationLoader& operator=(const AnimationLoader&) = delete;
    AnimationLoader& operator=(const AnimationLoader&&) = delete;

    private:
    AnimationLoader() = default;
    ~AnimationLoader() = default;
};
}  // namespace spark::loaders