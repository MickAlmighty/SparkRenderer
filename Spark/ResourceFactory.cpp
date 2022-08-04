#include "ResourceFactory.h"

#include "Resource.h"
#include "ResourceIdentifier.h"
#include "utils/CommonUtils.h"
#include "loaders/AnimationLoader.hpp"
#include "loaders/GltfLoader.hpp"
#include "loaders/ModelLoader.hpp"
#include "loaders/ShaderLoader.hpp"
#include "loaders/SceneLoader.hpp"
#include "loaders/TextureLoader.hpp"
#include "Logging.h"

namespace spark::resourceManagement
{
namespace sl = spark::loaders;

std::vector<std::unique_ptr<const sl::ILoader>> make_unique_loaders() {
    std::vector<std::unique_ptr<const sl::ILoader>> l;
    l.emplace_back(std::make_unique<const sl::ModelLoader>());
    l.emplace_back(std::make_unique<const sl::GltfLoader>());
    l.emplace_back(std::make_unique<const sl::ShaderLoader>());
    l.emplace_back(std::make_unique<const sl::TextureLoader>());
    l.emplace_back(std::make_unique<const sl::SceneLoader>());
    l.emplace_back(std::make_unique<const sl::AnimationLoader>());
    return l;
}

const std::vector<std::unique_ptr<const sl::ILoader>> loaders = make_unique_loaders();

std::shared_ptr<Resource> ResourceFactory::loadResource(const std::filesystem::path& resourcesRootPath,
                                                        const std::filesystem::path& resourceRelativePath)
{
    const auto ext = extensionToLowerCase(resourceRelativePath.string());

    for each(const auto& loader in loaders)
    {
        if(loader->isExtensionSupported(ext))
        {
            return loader->load(resourcesRootPath, resourceRelativePath);
        }
    }

    SPARK_ERROR("File could not be loaded! {}", resourceRelativePath.string());
    return nullptr;
}

bool ResourceFactory::isExtensionSupported(const std::filesystem::path& filePath)
{
    const auto ext = extensionToLowerCase(filePath);
    for each(const auto& loader in loaders)
    {
        if(loader->isExtensionSupported(ext))
        {
            return true;
        }
    }

    SPARK_WARN("Extension not supported! {}", filePath.string());
    return false;
}

std::string ResourceFactory::extensionToLowerCase(const std::filesystem::path& path)
{
    return spark::utils::toLowerCase(path.extension().string());
}

std::vector<std::string> ResourceFactory::supportedAnimationExtensions()
{
    return loaders::AnimationLoader().supportedExtensions();
}

std::vector<std::string> ResourceFactory::supportedModelExtensions()
{
    auto modelExtensions = sl::ModelLoader().supportedExtensions();
    auto gltfExtensions = sl::GltfLoader().supportedExtensions();

    std::vector<std::string> exts;
    exts.reserve(modelExtensions.size() + gltfExtensions.size());

    std::move(modelExtensions.begin(), modelExtensions.end(), std::back_inserter(exts));
    std::move(gltfExtensions.begin(), gltfExtensions.end(), std::back_inserter(exts));

    return exts;
}

std::vector<std::string> ResourceFactory::supportedTextureExtensions()
{
    return sl::TextureLoader().supportedExtensions();
}

std::vector<std::string> ResourceFactory::supportedShaderExtensions()
{
    return sl::ShaderLoader().supportedExtensions();
}

std::vector<std::string> ResourceFactory::supportedSceneExtensions()
{
    return sl::SceneLoader().supportedExtensions();
}

std::vector<std::string> ResourceFactory::supportedExtensions()
{
    std::vector<std::string> all;
    all.reserve(16);

    for(const auto& loader : loaders)
    {
        const auto exts = loader->supportedExtensions();
        std::move(exts.begin(), exts.end(), std::back_inserter(all));
    }
    return all;
}
}  // namespace spark::resourceManagement
