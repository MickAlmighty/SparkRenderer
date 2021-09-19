#include "ResourceLibrary.h"

#include "Logging.h"
#include "ResourceFactory.h"
#include "Spark.h"

namespace spark::resourceManagement
{
std::function<bool(const std::shared_ptr<ResourceIdentifier>&)> findWithinExtensions(const std::vector<std::string>& extensions)
{
    return [&extensions](const std::shared_ptr<resourceManagement::ResourceIdentifier>& resId) {
        return std::find(extensions.cbegin(), extensions.cend(), resId->getResourceExtensionLowerCase()) != extensions.end();
    };
}

ResourceLibrary::ResourceLibrary(const std::filesystem::path& pathToResources) : pathToResources(pathToResources)
{
    if(const auto cacheDir = pathToResources / "cache"; !std::filesystem::exists(cacheDir))
    {
        std::filesystem::create_directory(cacheDir);
    }

    createResourceIdentifiers();
}

ResourceLibrary::~ResourceLibrary()
{
    resourceIdentifiers.clear();
}

std::vector<std::shared_ptr<ResourceIdentifier>> ResourceLibrary::getModelResourceIdentifiers() const
{
    return getResourceIdentifiers(findWithinExtensions(ResourceFactory::supportedModelExtensions()));
}

std::vector<std::shared_ptr<ResourceIdentifier>> ResourceLibrary::getTextureResourceIdentifiers() const
{
    return getResourceIdentifiers(findWithinExtensions(ResourceFactory::supportedTextureExtensions()));
}

std::vector<std::shared_ptr<ResourceIdentifier>> ResourceLibrary::getShaderResourceIdentifiers() const
{
    return getResourceIdentifiers(findWithinExtensions(ResourceFactory::supportedShaderExtensions()));
}

std::vector<std::shared_ptr<ResourceIdentifier>> ResourceLibrary::getSceneResourceIdentifiers() const
{
    return getResourceIdentifiers(findWithinExtensions(ResourceFactory::supportedSceneExtensions()));
}

void ResourceLibrary::createResourceIdentifiers()
{
    for(const auto& directory_entry : std::filesystem::recursive_directory_iterator(pathToResources))
    {
        if(ResourceFactory::isExtensionSupported(directory_entry.path()))
        {
            const auto relativePath = directory_entry.path().lexically_relative(pathToResources);
            resourceIdentifiers.insert(std::make_shared<ResourceIdentifier>(pathToResources, relativePath));
        }
    }
}

std::vector<std::shared_ptr<ResourceIdentifier>> ResourceLibrary::getResourceIdentifiers() const
{
    std::vector<std::shared_ptr<ResourceIdentifier>> resIds;
    resIds.reserve(resourceIdentifiers.size());

    for(const auto& resourceIdentifier : resourceIdentifiers)
    {
        resIds.push_back(resourceIdentifier);
    }

    return resIds;
}

std::vector<std::shared_ptr<ResourceIdentifier>> ResourceLibrary::getResourceIdentifiers(
    const std::function<bool(const std::shared_ptr<ResourceIdentifier>&)>& comp) const
{
    return filter(resourceIdentifiers.begin(), resourceIdentifiers.end(), comp);
}

std::shared_ptr<ResourceIdentifier> ResourceLibrary::getResourceIdentifier(const std::filesystem::path& path)
{
    const auto findByPath = [&path](const std::shared_ptr<ResourceIdentifier>& ri) { return ri->getFullPath() == path; };

    const auto it = std::find_if(resourceIdentifiers.begin(), resourceIdentifiers.end(), findByPath);

    if(it != resourceIdentifiers.end())
    {
        return *it;
    }

    return nullptr;
}
}  // namespace spark::resourceManagement
