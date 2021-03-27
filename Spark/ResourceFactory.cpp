#include "ResourceFactory.h"

#include "Model.h"
#include "Resource.h"
#include "ResourceIdentifier.h"
#include "Shader.h"
#include "Texture.h"

namespace spark::resourceManagement
{
std::map<std::filesystem::path, std::function<std::shared_ptr<Resource>(const ResourceIdentifier& id)>> ResourceFactory::resourceCreationFunctions{
    // TODO: replace with a reflection-based list
    {".obj", [](const ResourceIdentifier& id) { return std::make_shared<resources::Model>(id); }},
    {".fbx", [](const ResourceIdentifier& id) { return std::make_shared<resources::Model>(id); }},
    {".FBX", [](const ResourceIdentifier& id) { return std::make_shared<resources::Model>(id); }},

    {".dds", [](const ResourceIdentifier& id) { return std::make_shared<resources::Texture>(id); }},
    {".DDS", [](const ResourceIdentifier& id) { return std::make_shared<resources::Texture>(id); }},
    {".ktx", [](const ResourceIdentifier& id) { return std::make_shared<resources::Texture>(id); }},
    {".KTX", [](const ResourceIdentifier& id) { return std::make_shared<resources::Texture>(id); }},
    {".KTX", [](const ResourceIdentifier& id) { return std::make_shared<resources::Texture>(id); }},
    {".png", [](const ResourceIdentifier& id) { return std::make_shared<resources::Texture>(id); }},
    {".jpg", [](const ResourceIdentifier& id) { return std::make_shared<resources::Texture>(id); }},
    {".tga", [](const ResourceIdentifier& id) { return std::make_shared<resources::Texture>(id); }},

    {".glsl", [](const ResourceIdentifier& id) { return std::make_shared<resources::Shader>(id); }},
};

std::optional<std::shared_ptr<Resource>> ResourceFactory::createResource(const std::filesystem::path& filePath)
{
    const auto it = resourceCreationFunctions.find(filePath.extension());
    if(it != resourceCreationFunctions.end())
    {
        const ResourceIdentifier id(filePath);
        const auto [extension, resourceCreationFunction] = *it;
        return resourceCreationFunction(id);
    }

    return std::nullopt;
}

}  // namespace spark::resourceManagement
