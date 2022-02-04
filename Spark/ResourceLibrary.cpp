#include "ResourceLibrary.h"

#include "Logging.h"
#include "ResourceFactory.h"
#include "Spark.h"

namespace spark::resourceManagement
{
std::function<bool(const std::shared_ptr<ResourceIdentifier>&)> findWithinExtensions(const std::vector<std::string>& extensions)
{
    return [&extensions](const std::shared_ptr<resourceManagement::ResourceIdentifier>& resId)
    { return std::find(extensions.cbegin(), extensions.cend(), resId->getResourceExtensionLowerCase()) != extensions.end(); };
}

ResourceLibrary::ResourceLibrary(const std::filesystem::path& pathToResources) : resourcesRootPath(pathToResources)
{
    if(const auto cacheDir = resourcesRootPath / "cache"; !std::filesystem::exists(cacheDir))
    {
        std::filesystem::create_directory(cacheDir);
    }
}

ResourceLibrary::~ResourceLibrary()
{
    resourceIdentifiers.clear();
}

void ResourceLibrary::cleanResourceRegistry()
{
    for(auto it = resourceIdentifiers.begin(); it != resourceIdentifiers.end();)
    {
        if(!it->get()->isResourceInUse())
        {
            it = resourceIdentifiers.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

std::filesystem::path ResourceLibrary::getResourcesRootPath() const
{
    return resourcesRootPath;
}
}  // namespace spark::resourceManagement
