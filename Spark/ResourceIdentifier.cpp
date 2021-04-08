#include "ResourceIdentifier.h"

#include <cstdio>
#include <iostream>

#include "Logging.h"
#include "ResourceFactory.h"

namespace spark::resourceManagement
{
ResourceIdentifier::ResourceIdentifier(const std::filesystem::path& fullResourcePath)
{
    if(!std::filesystem::exists(fullResourcePath))
    {
        SPARK_CRITICAL("File does not exist");
        throw std::runtime_error("File does not exist");
    }

    if(!fullResourcePath.has_filename())
    {
        SPARK_CRITICAL("Trying to create resource identifier without filename");
        throw std::runtime_error("Trying to create resource identifier without filename");
    }

    if(!fullResourcePath.has_extension())
    {
        SPARK_CRITICAL("Trying to create resource identifier with name without extension");
        throw std::runtime_error("Trying to create resource identifier with name without extension");
    }

    if(!fullResourcePath.has_parent_path())
    {
        SPARK_CRITICAL("Trying to create resource identifier without parent path");
        throw std::runtime_error("Trying to create resource identifier without parent path");
    }

    resourcePath = fullResourcePath;
}

ResourceIdentifier::ResourceIdentifier(const ResourceIdentifier&& identifier) noexcept : resourcePath(std::move(identifier.resourcePath)) {}

bool ResourceIdentifier::operator==(const ResourceIdentifier& identifier) const
{
    return resourcePath == identifier.resourcePath;
}

bool ResourceIdentifier::operator<(const ResourceIdentifier& identifier) const
{
    return resourcePath < identifier.resourcePath;
}

std::filesystem::path ResourceIdentifier::getFullPath() const
{
    return resourcePath;
}

std::filesystem::path ResourceIdentifier::getDirectoryPath() const
{
    return resourcePath.parent_path();
}

std::filesystem::path ResourceIdentifier::getResourceName(bool withExtension) const
{
    if(withExtension)
    {
        return resourcePath.filename();
    }
    else
    {
        return resourcePath.stem();
    }
}

std::filesystem::path ResourceIdentifier::getResourceExtension() const
{
    return resourcePath.extension();
}

bool ResourceIdentifier::changeResourceDirectory(const std::filesystem::path& path)
{
    if(!std::filesystem::is_directory(path))
    {
        return false;
    }

    const auto directoryPath = path / resourcePath.filename();

    if(std::filesystem::exists(directoryPath))
    {
        SPARK_INFO("File with the same name already exists in this directory! Moving file aborted!");
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(resourcePath, directoryPath, ec);

    if(ec.value())
    {
        SPARK_ERROR(ec.message());
        return false;
    }

    resourcePath = directoryPath;
    return true;
}

bool ResourceIdentifier::changeResourceName(const std::filesystem::path& name)
{
    if(name.has_parent_path())
    {
        return false;
    }

    if(!name.has_extension())
    {
        return false;
    }

    const auto directoryPath = resourcePath.parent_path() / name;

    if(std::filesystem::exists(directoryPath))
    {
        SPARK_INFO("File does already exists! File rename aborted!");
        return false;
    }

    std::error_code ec;
    std::filesystem::rename(resourcePath, directoryPath, ec);

    if(ec.value())
    {
        SPARK_ERROR(ec.message());
    }
    else
    {
        if(std::filesystem::exists(directoryPath))
        {
            resourcePath = directoryPath;
            return true;
        }
    }

    return false;
}

std::shared_ptr<Resource> ResourceIdentifier::getResource()
{
    if(resource.expired())
    {
        const auto resourceOpt = ResourceFactory::createResource(getFullPath());
        if(resourceOpt.has_value())
        {
            resource = resourceOpt.value();
            return resource.lock();
        }
    }

    return resource.lock();
}
}  // namespace spark::resourceManagement
