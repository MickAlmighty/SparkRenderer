#include "ResourceLibrary.h"

#include "Logging.h"
#include "ResourceFactory.h"
#include "Spark.h"

namespace spark::resourceManagement
{
void ResourceLibrary::setup() {}

void ResourceLibrary::cleanup() {}

void ResourceLibrary::createResources(const std::filesystem::path& pathToResources)
{
    for(const auto& path : std::filesystem::recursive_directory_iterator(pathToResources))
    {
        if (ResourceFactory::isExtensionSupported(path))
        {
            resourceIdentifiers.insert(std::make_shared<ResourceIdentifier>(path));
        }
    }
}

std::vector<std::shared_ptr<ResourceIdentifier>> ResourceLibrary::getResourceIdentifiers() const
{
    std::vector<std::shared_ptr<ResourceIdentifier>> resIds;
    resIds.reserve(resourceIdentifiers.size());

    for (auto resourceIt = resourceIdentifiers.begin(); resourceIt != resourceIdentifiers.end(); ++resourceIt)
    {
        resIds.push_back(*resourceIt);
    }

    return resIds;
}

std::vector<std::shared_ptr<ResourceIdentifier>> ResourceLibrary::getResourceIdentifiers(const std::function<bool(const std::shared_ptr<ResourceIdentifier>&)>& comp) const
{
    return filter(resourceIdentifiers.begin(), resourceIdentifiers.end(), comp);
}
}  // namespace spark::resourceManagement
