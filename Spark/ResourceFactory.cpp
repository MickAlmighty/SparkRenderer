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
using path = std::filesystem::path;
namespace loaders = spark::loaders;

std::map<std::string, std::function<std::shared_ptr<Resource>(const path& resourcesRootPath, const path& resourceRelativePath)>>
    ResourceFactory::resourceCreationFunctions{
        // TODO: replace with a reflection-based list
        {".anim", [](const path& rootPath, const path& relativePath) { return loaders::AnimationLoader::load(rootPath, relativePath); }},
        {".obj", [](const path& rootPath, const path& relativePath) { return loaders::ModelLoader::load(rootPath, relativePath); }},
        {".fbx", [](const path& rootPath, const path& relativePath) { return loaders::ModelLoader::load(rootPath, relativePath); }},
        {".gltf", [](const path& rootPath, const path& relativePath) { return loaders::GltfLoader::load(rootPath, relativePath); }},
        {".dds", [](const path& rootPath, const path& relativePath) { return loaders::TextureLoader::load(rootPath, relativePath); }},
        {".ktx", [](const path& rootPath, const path& relativePath) { return loaders::TextureLoader::load(rootPath, relativePath); }},
        {".png", [](const path& rootPath, const path& relativePath) { return loaders::TextureLoader::load(rootPath, relativePath); }},
        {".tga", [](const path& rootPath, const path& relativePath) { return loaders::TextureLoader::load(rootPath, relativePath); }},
        {".jpg", [](const path& rootPath, const path& relativePath) { return loaders::TextureLoader::load(rootPath, relativePath); }},
        {".hdr", [](const path& rootPath, const path& relativePath) { return loaders::TextureLoader::load(rootPath, relativePath); }},
        {".glsl", [](const path& rootPath, const path& relativePath) { return loaders::ShaderLoader::load(rootPath, relativePath); }},
        {".scene", [](const path& rootPath, const path& relativePath) { return loaders::SceneLoader::load(rootPath, relativePath); }},
    };

std::shared_ptr<Resource> ResourceFactory::loadResource(const std::filesystem::path& resourcesRootPath,
                                                        const std::filesystem::path& resourceRelativePath)
{
    //const auto it = resourceCreationFunctions.find(extensionToLowerCase(resourceRelativePath.string()));
    //if(it != resourceCreationFunctions.end())
    //{
    //    const auto [extension, resourceCreationFunction] = *it;
    //    return resourceCreationFunction(resourcesRootPath, resourceRelativePath);
    //}


    const auto ext = extensionToLowerCase(resourceRelativePath.string());
    if (loaders::TextureLoader::isExtensionSupported(ext))
    {
        return loaders::TextureLoader::load(resourcesRootPath, resourceRelativePath);
    }
    else if(loaders::ModelLoader::isExtensionSupported(ext))
    {
        return loaders::ModelLoader::load(resourcesRootPath, resourceRelativePath);
    }
    else if(loaders::ShaderLoader::isExtensionSupported(ext))
    {
        return loaders::ShaderLoader::load(resourcesRootPath, resourceRelativePath);
    }
    else if(loaders::GltfLoader::isExtensionSupported(ext))
    {
        return loaders::GltfLoader::load(resourcesRootPath, resourceRelativePath);
    }
    else if(loaders::SceneLoader::isExtensionSupported(ext))
    {
        return loaders::SceneLoader::load(resourcesRootPath, resourceRelativePath);
    }
    else if(loaders::AnimationLoader::isExtensionSupported(ext))
    {
        return loaders::AnimationLoader::load(resourcesRootPath, resourceRelativePath);
    }

    SPARK_ERROR("File could not be loaded! {}", resourceRelativePath.string());
    return nullptr;
}

bool ResourceFactory::isExtensionSupported(const std::filesystem::path& filePath)
{
    return resourceCreationFunctions.find(extensionToLowerCase(filePath)) != resourceCreationFunctions.end();
}

std::string ResourceFactory::extensionToLowerCase(const std::filesystem::path& path)
{
    return spark::utils::toLowerCase(path.extension().string());
}

std::vector<std::string> ResourceFactory::supportedAnimationExtensions()
{
    return loaders::AnimationLoader::supportedExtensions();
}

std::vector<std::string> ResourceFactory::supportedModelExtensions()
{
    auto modelExtensions = loaders::ModelLoader::supportedExtensions();
    auto gltfExtensions = loaders::GltfLoader::supportedExtensions();

    std::vector<std::string> exts;
    exts.reserve(modelExtensions.size() + gltfExtensions.size());

    std::move(modelExtensions.begin(), modelExtensions.end(), std::back_inserter(exts));
    std::move(gltfExtensions.begin(), gltfExtensions.end(), std::back_inserter(exts));

    return exts;
}

std::vector<std::string> ResourceFactory::supportedTextureExtensions()
{
    return loaders::TextureLoader::supportedExtensions();
}

std::vector<std::string> ResourceFactory::supportedShaderExtensions()
{
    return loaders::ShaderLoader::supportedExtensions();
}

std::vector<std::string> ResourceFactory::supportedSceneExtensions()
{
    return loaders::SceneLoader::supportedExtensions();
}

std::vector<std::string> ResourceFactory::supportedExtensions()
{
    std::vector<std::string> all;
    all.reserve(resourceCreationFunctions.size());

    for(const auto& [supportedExtension, resourceLoader] : resourceCreationFunctions)
    {
        all.push_back(supportedExtension);
    }
    return all;
}
}  // namespace spark::resourceManagement
