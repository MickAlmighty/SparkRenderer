#include "ResourceFactory.h"

#include "GltfLoader.hpp"
#include "Resource.h"
#include "ResourceIdentifier.h"
#include "ResourceLoader.h"
#include "Shader.h"

namespace spark::resourceManagement
{
std::map<std::string, std::function<std::shared_ptr<Resource>(const std::shared_ptr<ResourceIdentifier>& ri)>>
    ResourceFactory::resourceCreationFunctions{
        // TODO: replace with a reflection-based list
        {".obj", [](const std::shared_ptr<ResourceIdentifier>& ri) { return ResourceLoader::createModel(ri); }},
        {".fbx", [](const std::shared_ptr<ResourceIdentifier>& ri) { return ResourceLoader::createModel(ri); }},
        {".gltf", [](const std::shared_ptr<ResourceIdentifier>& ri) { return GltfLoader().createModel(ri); }},
        {".dds", [](const std::shared_ptr<ResourceIdentifier>& ri) { return ResourceLoader::createCompressedTexture(ri); }},
        {".ktx", [](const std::shared_ptr<ResourceIdentifier>& ri) { return ResourceLoader::createCompressedTexture(ri); }},
        {".png", [](const std::shared_ptr<ResourceIdentifier>& ri) { return ResourceLoader::createUncompressedTexture(ri); }},
        {".jpg", [](const std::shared_ptr<ResourceIdentifier>& ri) { return ResourceLoader::createUncompressedTexture(ri); }},
        {".tga", [](const std::shared_ptr<ResourceIdentifier>& ri) { return ResourceLoader::createUncompressedTexture(ri); }},
        {".hdr", [](const std::shared_ptr<ResourceIdentifier>& ri) { return ResourceLoader::createHdrTexture(ri); }},
        {".glsl", [](const std::shared_ptr<ResourceIdentifier>& ri) { return std::make_shared<resources::Shader>(ri->getFullPath()); }},
        {".scene", [](const std::shared_ptr<ResourceIdentifier>& ri) { return ResourceLoader::createScene(ri); }},
    };

std::shared_ptr<Resource> ResourceFactory::loadResource(const std::shared_ptr<ResourceIdentifier>& resourceIdentifier)
{
    const auto it = resourceCreationFunctions.find(extensionToLowerCase(resourceIdentifier->getRelativePath().string()));
    if(it != resourceCreationFunctions.end())
    {
        const auto [extension, resourceCreationFunction] = *it;
        return resourceCreationFunction(resourceIdentifier);
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
