#include "ResourceFactory.h"

#include "Resource.h"
#include "ResourceLoader.h"
#include "Shader.h"

namespace spark::resourceManagement
{
std::map<std::filesystem::path, std::function<std::shared_ptr<Resource>(const std::filesystem::path& path)>>
    ResourceFactory::resourceCreationFunctions{
        // TODO: replace with a reflection-based list
        {".obj", [](const std::filesystem::path& path) { return ResourceLoader::createModel(path); }},
        {".fbx", [](const std::filesystem::path& path) { return ResourceLoader::createModel(path); }},
        {".FBX", [](const std::filesystem::path& path) { return ResourceLoader::createModel(path); }},

        {".dds", [](const std::filesystem::path& path) { return ResourceLoader::createCompressedTexture(path); }},
        {".DDS", [](const std::filesystem::path& path) { return ResourceLoader::createCompressedTexture(path); }},
        {".ktx", [](const std::filesystem::path& path) { return ResourceLoader::createCompressedTexture(path); }},
        {".KTX", [](const std::filesystem::path& path) { return ResourceLoader::createCompressedTexture(path); }},

        {".png", [](const std::filesystem::path& path) { return ResourceLoader::createUncompressedTexture(path); }},
        {".jpg", [](const std::filesystem::path& path) { return ResourceLoader::createUncompressedTexture(path); }},
        {".tga", [](const std::filesystem::path& path) { return ResourceLoader::createUncompressedTexture(path); }},

        {".glsl", [](const std::filesystem::path& path) { return std::make_shared<resources::Shader>(path); }},
    };

std::shared_ptr<Resource> ResourceFactory::createResource(const std::filesystem::path& filePath)
{
    const auto it = resourceCreationFunctions.find(filePath.extension());
    if(it != resourceCreationFunctions.end())
    {
        const auto [extension, resourceCreationFunction] = *it;
        return resourceCreationFunction(filePath);
    }

    return nullptr;
}

bool ResourceFactory::isExtensionSupported(const std::filesystem::path& filePath)
{
    return resourceCreationFunctions.find(filePath.extension()) != resourceCreationFunctions.end();
}
}  // namespace spark::resourceManagement
