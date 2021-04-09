#include "ResourceLibrary.h"

#include "Logging.h"
#include "ResourceFactory.h"
#include "Spark.h"

namespace spark::resourceManagement
{
void ResourceLibrary::setup(const std::filesystem::path& pathToResources)
{
    createResources(pathToResources);
}

void ResourceLibrary::cleanup()
{
    resourceIdentifiers.clear();
}

std::function<bool(const std::shared_ptr<ResourceIdentifier>&)> findWithinExtensions(const std::vector<std::string>& extensions)
{
    return [&extensions](const std::shared_ptr<resourceManagement::ResourceIdentifier>& resId) {
        return std::find(extensions.cbegin(), extensions.cend(), resId->getResourceExtension().string()) != extensions.end();
    };
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

void ResourceLibrary::createResources(const std::filesystem::path& pathToResources)
{
    for(const auto& path : std::filesystem::recursive_directory_iterator(pathToResources))
    {
        if(ResourceFactory::isExtensionSupported(path))
        {
            resourceIdentifiers.insert(std::make_shared<ResourceIdentifier>(path));
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
}  // namespace spark::resourceManagement
