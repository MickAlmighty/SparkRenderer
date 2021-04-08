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
    ResourceLibrary() = default;
    ~ResourceLibrary() = default;

    ResourceLibrary(const ResourceLibrary&) = delete;
    ResourceLibrary(const ResourceLibrary&&) = delete;
    ResourceLibrary operator=(const ResourceLibrary&) = delete;
    ResourceLibrary operator=(const ResourceLibrary&&) = delete;

    void setup(const std::filesystem::path& pathToResources);
    void cleanup();

    std::vector<std::shared_ptr<ResourceIdentifier>> getResourceIdentifiers() const;
    std::vector<std::shared_ptr<ResourceIdentifier>> getResourceIdentifiers(
        const std::function<bool(const std::shared_ptr<ResourceIdentifier>&)>& comp) const;

    template<typename T>
    std::shared_ptr<T> getResourceByName(const std::string& resourceName) const;

    template<typename T>
    std::shared_ptr<T> getResourceByPath(const std::string& resourcePath) const;

    template<typename T>
    std::shared_ptr<T> getResource(const std::function<bool(const std::shared_ptr<ResourceIdentifier>& resourceIdentifier)>& searchFunc) const;

    private:
    template<class InputIterator, class Functor>
    [[nodiscard]] std::vector<typename std::iterator_traits<InputIterator>::value_type> static filter(const InputIterator& begin,
                                                                                                      const InputIterator& end, Functor f);
    void createResources(const std::filesystem::path& pathToResources);

    std::set<std::shared_ptr<ResourceIdentifier>> resourceIdentifiers;
};

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResourceByName(const std::string& resourceName) const
{
    const auto findByName = [&resourceName](const std::shared_ptr<ResourceIdentifier>& resourceIdentifier) {
        const bool validName = resourceIdentifier->getResourceName().string() == resourceName;
        return validName;
    };

    return getResource<T>(findByName);
}

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResourceByPath(const std::string& resourcePath) const
{
    const auto findByPath = [&resourcePath](const std::shared_ptr<ResourceIdentifier>& resourceIdentifier) {
        const bool validName = resourceIdentifier->getFullPath().string() == resourcePath;
        return validName;
    };

    return getResource<T>(findByPath);
}

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResource(
    const std::function<bool(const std::shared_ptr<ResourceIdentifier>& resourceIdentifier)>& searchFunc) const
{
    const auto resId_it = std::find_if(resourceIdentifiers.begin(), resourceIdentifiers.end(), searchFunc);

    if(resId_it != resourceIdentifiers.end())
    {
        return std::dynamic_pointer_cast<T>((*resId_it)->getResource());
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