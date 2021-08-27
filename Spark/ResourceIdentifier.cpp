#include "ResourceIdentifier.h"

#include <iostream>
#include <utility>

#include "Logging.h"
#include "ResourceFactory.h"

namespace spark::resourceManagement
{
ResourceIdentifier::ResourceIdentifier(std::filesystem::path pathToResources, std::filesystem::path relativePath)
    : pathToResources(std::move(pathToResources)), relativePathToResource(std::move(relativePath))
{
}

bool ResourceIdentifier::operator==(const ResourceIdentifier& identifier) const
{
    return relativePathToResource == identifier.relativePathToResource;
}

bool ResourceIdentifier::operator<(const ResourceIdentifier& identifier) const
{
    return relativePathToResource < identifier.relativePathToResource;
}

std::filesystem::path ResourceIdentifier::getFullPath() const
{
    return pathToResources / relativePathToResource;
}

std::filesystem::path ResourceIdentifier::getRelativePath() const
{
    return relativePathToResource;
}

std::filesystem::path ResourceIdentifier::getResourceName(bool withExtension) const
{
    if(withExtension)
    {
        return relativePathToResource.filename();
    }

    return relativePathToResource.stem();
}

std::string ResourceIdentifier::getResourceExtension() const
{
    return relativePathToResource.extension().string();
}

std::string ResourceIdentifier::getResourceExtensionLowerCase() const
{
    return extensionToLowerCase(relativePathToResource);
}

std::shared_ptr<Resource> ResourceIdentifier::getResource()
{
    if(resource.expired())
    {
        if(const auto resourcePtr = ResourceFactory::loadResource(shared_from_this()); resourcePtr)
        {
            resource = resourcePtr;
            return resource.lock();
        }
    }

    return resource.lock();
}

std::string ResourceIdentifier::extensionToLowerCase(const std::filesystem::path& path) const
{
    std::string ext = path.extension().string();
    std::for_each(ext.begin(), ext.end(), [](char& c) { c = std::tolower(c); });

    return ext;
}
}  // namespace spark::resourceManagement
