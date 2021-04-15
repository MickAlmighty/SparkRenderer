#include "ResourceFactory.h"

#include "Resource.h"
#include "ResourceLoader.h"
#include "Shader.h"

namespace spark::resourceManagement
{
std::map<std::string, std::function<std::shared_ptr<Resource>(const std::filesystem::path& path)>> ResourceFactory::resourceCreationFunctions{
    // TODO: replace with a reflection-based list
    {".obj", [](const std::filesystem::path& path) { return ResourceLoader::createModel(path); }},
    {".fbx", [](const std::filesystem::path& path) { return ResourceLoader::createModel(path); }},
    {".dds", [](const std::filesystem::path& path) { return ResourceLoader::createCompressedTexture(path); }},
    {".ktx", [](const std::filesystem::path& path) { return ResourceLoader::createCompressedTexture(path); }},
    {".png", [](const std::filesystem::path& path) { return ResourceLoader::createUncompressedTexture(path); }},
    {".jpg", [](const std::filesystem::path& path) { return ResourceLoader::createUncompressedTexture(path); }},
    {".tga", [](const std::filesystem::path& path) { return ResourceLoader::createUncompressedTexture(path); }},
    {".hdr", [](const std::filesystem::path& path) { return ResourceLoader::createHdrTexture(path); }},
    {".glsl", [](const std::filesystem::path& path) { return std::make_shared<resources::Shader>(path); }},
    {".scene", [](const std::filesystem::path& path) { return ResourceLoader::createScene(path); }},
};

std::shared_ptr<Resource> ResourceFactory::loadResource(const std::filesystem::path& filePath)
{
    const auto it = resourceCreationFunctions.find(extensionToLowerCase(filePath));
    if(it != resourceCreationFunctions.end())
    {
        const auto [extension, resourceCreationFunction] = *it;
        return resourceCreationFunction(filePath);
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
    std::for_each(ext.begin(), ext.end(), [](char& c) { c = std::tolower(c); });

    return ext;
}

std::vector<std::string> ResourceFactory::supportedModelExtensions()
{
    return std::vector<std::string>{".obj", ".fbx"};
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
