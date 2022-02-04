#pragma once

#include <filesystem>
#include <functional>
#include <set>

#include "ResourceFactory.h"
#include "ResourceIdentifier.h"

namespace spark::resourceManagement
{
class ResourceLibrary
{
    public:
    ResourceLibrary(const std::filesystem::path& pathToResources);
    ~ResourceLibrary();

    ResourceLibrary(const ResourceLibrary&) = delete;
    ResourceLibrary(const ResourceLibrary&&) = delete;
    ResourceLibrary operator=(const ResourceLibrary&) = delete;
    ResourceLibrary operator=(const ResourceLibrary&&) = delete;

    void cleanResourceRegistry();

    std::filesystem::path getResourcesRootPath() const;

    template<typename T>
    std::shared_ptr<T> getResourceByFullPath(const std::filesystem::path& resourcePath);

    template<typename T>
    std::shared_ptr<T> getResourceByRelativePath(const std::filesystem::path& resourcePath);

    template<typename T>
    std::shared_ptr<T> getResource(const std::function<bool(const std::shared_ptr<ResourceIdentifier>& resourceIdentifier)>& searchFunc) const;

    private:
    template<class InputIterator, class Functor>
    [[nodiscard]] std::vector<typename std::iterator_traits<InputIterator>::value_type> static filter(const InputIterator& begin,
                                                                                                      const InputIterator& end, Functor f);

    std::set<std::shared_ptr<ResourceIdentifier>> resourceIdentifiers;
    const std::filesystem::path resourcesRootPath;
};

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResourceByFullPath(const std::filesystem::path& resourcePath)
{
    const auto findByPath = [&resourcePath](const std::shared_ptr<ResourceIdentifier>& resourceIdentifier)
    { return resourceIdentifier->getFullPath() == resourcePath; };

    if(const auto resource = getResource<T>(findByPath); resource)
    {
        return resource;
    }

    if(std::filesystem::exists(resourcePath))
    {
        const auto ri = std::make_shared<ResourceIdentifier>(resourcesRootPath, resourcePath.lexically_relative(resourcesRootPath));
        resourceIdentifiers.insert(ri);

        return std::static_pointer_cast<T>(ri->getResource());
    }

    return nullptr;
}

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResourceByRelativePath(const std::filesystem::path& resourcePath)
{
    const auto findByPath = [&resourcePath](const std::shared_ptr<ResourceIdentifier>& resourceIdentifier)
    { return resourceIdentifier->getRelativePath().string() == resourcePath; };

    if(const auto resource = getResource<T>(findByPath); resource)
    {
        return resource;
    }

    if(std::filesystem::exists(resourcesRootPath / resourcePath))
    {
        const auto ri = std::make_shared<ResourceIdentifier>(resourcesRootPath, resourcePath);
        resourceIdentifiers.insert(ri);

        return std::static_pointer_cast<T>(ri->getResource());
    }

    return nullptr;
}

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResource(
    const std::function<bool(const std::shared_ptr<ResourceIdentifier>& resourceIdentifier)>& searchFunc) const
{
    const auto resId_it = std::find_if(resourceIdentifiers.begin(), resourceIdentifiers.end(), searchFunc);

    if(resId_it != resourceIdentifiers.end())
    {
        return std::static_pointer_cast<T>((*resId_it)->getResource());
    }

    return nullptr;
}

template<class InputIterator, class Functor>
std::vector<typename std::iterator_traits<InputIterator>::value_type> ResourceLibrary::filter(const InputIterator& begin, const InputIterator& end,
                                                                                              Functor f)
{
    using ValueType = typename std::iterator_traits<InputIterator>::value_type;

    std::vector<ValueType> result;
    result.reserve(std::distance(begin, end));

    std::copy_if(begin, end, std::back_inserter(result), f);

    return result;
}
}  // namespace spark::resourceManagement