#include "SceneLoader.hpp"

#include "JsonSerializer.h"
#include "Resource.h"
#include "Scene.h"

namespace spark::loaders
{
std::shared_ptr<resourceManagement::Resource> SceneLoader::load(const std::filesystem::path& resourcesRootPath,
                                                                const std::filesystem::path& resourceRelativePath) const
{
    const auto path = resourcesRootPath / resourceRelativePath;
    if(auto scene = JsonSerializer().loadSceneFromFile(path); scene)
    {
        scene->setPath(resourceRelativePath);
        return scene;
    }

    return nullptr;
}

bool SceneLoader::isExtensionSupported(const std::string& ext) const
{
    const auto supportedExts = supportedExtensions();
    const auto it = std::find_if(supportedExts.begin(), supportedExts.end(), [&ext](const auto& e) { return e == ext; });
    return it != supportedExts.end();
}

std::vector<std::string> SceneLoader::supportedExtensions() const
{
    return {".scene"};
}
}  // namespace spark::loaders
