#include "AnimationLoader.hpp"

#include "AnimationData.hpp"
#include "JsonSerializer.h"

namespace spark::loaders
{
std::shared_ptr<resourceManagement::Resource> AnimationLoader::load(const std::filesystem::path& resourcesRootPath,
                                                                    const std::filesystem::path& resourceRelativePath)
{
    const auto path = resourcesRootPath / resourceRelativePath;
    if(const auto animation = JsonSerializer().loadShared<resources::AnimationData>(path); animation)
    {
        animation->setPath(resourceRelativePath);
        return animation;
    }

    return nullptr;
}

bool AnimationLoader::isExtensionSupported(const std::string& ext)
{
    const auto supportedExts = supportedExtensions();
    const auto it = std::find_if(supportedExts.begin(), supportedExts.end(), [&ext](const auto& e) { return e == ext; });
    return it != supportedExts.end();
}

std::vector<std::string> AnimationLoader::supportedExtensions()
{
    return {".anim"};
}
}  // namespace spark::loaders