#include "ResourceFactory.h"

#include "GltfLoader.hpp"
#include "Resource.h"
#include "ResourceIdentifier.h"
#include "ResourceLoader.h"

namespace spark::resourceManagement
{

using path = std::filesystem::path;

std::map<std::string, std::function<std::shared_ptr<Resource>(const path& resourcesRootPath, const path& resourceRelativePath)>>
    ResourceFactory::resourceCreationFunctions{
        // TODO: replace with a reflection-based list
        {".anim",
         [](const path& resourcesRootPath, const path& relativePath) { return ResourceLoader::createAnimation(resourcesRootPath, relativePath); }},
        {".obj", [](const path& resourcesRootPath, const path& relativePath) { return ResourceLoader::createModel(resourcesRootPath, relativePath); }},
        {".fbx", [](const path& resourcesRootPath, const path& relativePath) { return ResourceLoader::createModel(resourcesRootPath, relativePath); }},
        {".gltf", [](const path& resourcesRootPath, const path& relativePath) { return GltfLoader().createModel(resourcesRootPath, relativePath); }},
        {".dds", [](const path& resourcesRootPath, const path& relativePath) { return ResourceLoader::createCompressedTexture(resourcesRootPath, relativePath); }},
        {".ktx", [](const path& resourcesRootPath, const path& relativePath) { return ResourceLoader::createCompressedTexture(resourcesRootPath, relativePath); }},
        {".png", [](const path& resourcesRootPath, const path& relativePath) { return ResourceLoader::createUncompressedTexture(resourcesRootPath, relativePath); }},
        {".jpg", [](const path& resourcesRootPath, const path& relativePath) { return ResourceLoader::createUncompressedTexture(resourcesRootPath, relativePath); }},
        {".tga", [](const path& resourcesRootPath, const path& relativePath) { return ResourceLoader::createUncompressedTexture(resourcesRootPath, relativePath); }},
        {".hdr", [](const path& resourcesRootPath, const path& relativePath) { return ResourceLoader::createHdrTexture(resourcesRootPath, relativePath); }},
        {".glsl", [](const path& resourcesRootPath, const path& relativePath) { return ResourceLoader::createShader(resourcesRootPath, relativePath); }},
        {".scene", [](const path& resourcesRootPath, const path& relativePath) { return ResourceLoader::createScene(resourcesRootPath, relativePath); }},
    };

std::shared_ptr<Resource> ResourceFactory::loadResource(const std::filesystem::path& resourcesRootPath,
                                                        const std::filesystem::path& resourceRelativePath)
{
    const auto it = resourceCreationFunctions.find(extensionToLowerCase(resourceRelativePath.string()));
    if(it != resourceCreationFunctions.end())
    {
        const auto [extension, resourceCreationFunction] = *it;
        return resourceCreationFunction(resourcesRootPath, resourceRelativePath);
    }

    return nullptr;
}

bool ResourceFactory::isExtensionSupported(const std::filesystem::path& filePath)
{
    return resourceCreationFunctions.find(extensionToLowerCase(filePath)) != resourceCreationFunctions.end();
}

std::string ResourceFactory::extensionToLowerCase(const std::filesystem::path& path)
{
    std::string ext = path.extension().string();
    std::for_each(ext.begin(), ext.end(), [](char& c) { c = static_cast<char>(std::tolower(c)); });

    return ext;
}

std::vector<std::string> ResourceFactory::supportedAnimationExtensions()
{
    return std::vector<std::string>{".anim"};
}

std::vector<std::string> ResourceFactory::supportedModelExtensions()
{
    return std::vector<std::string>{".obj", ".fbx", ".gltf"};
}

std::vector<std::string> ResourceFactory::supportedTextureExtensions()
{
    return std::vector<std::string>{".dds", ".ktx", ".png", ".jpg", ".tga", ".hdr"};
}

std::vector<std::string> ResourceFactory::supportedShaderExtensions()
{
    return std::vector<std::string>{".glsl"};
}

std::vector<std::string> ResourceFactory::supportedSceneExtensions()
{
    return std::vector<std::string>{".scene"};
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
