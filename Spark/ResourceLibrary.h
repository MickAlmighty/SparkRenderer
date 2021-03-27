#pragma once

#include <deque>
#include <filesystem>
#include <optional>
#include <set>

#include <object_threadsafe/safe_ptr.h>

#include "Resource.h"
#include "Spark.h"

namespace spark::resourceManagement
{
enum class ProcessingType
{
    LOAD,
    UNLOAD
};

struct ResourceProcessingInfo
{
    std::shared_ptr<Resource> resource = nullptr;
    ProcessingType type = ProcessingType::LOAD;
    bool success{false};
};

class ResourceLibrary
{
    public:
    ResourceLibrary() = default;
    ~ResourceLibrary() = default;

    ResourceLibrary(const ResourceLibrary& library) = delete;
    ResourceLibrary(const ResourceLibrary&& library) = delete;
    ResourceLibrary operator=(const ResourceLibrary& ResourceLibrary) = delete;
    ResourceLibrary operator=(const ResourceLibrary&& ResourceLibrary) = delete;

    void setup();
    void cleanup();
    void createResources(const std::filesystem::path& pathToResources);
    void processGpuResources();

    size_t getLoadedResourcesCount() const;

    template<typename T>
    std::vector<ResourceIdentifier> getResourceIdentifiers() const;

    template<typename T>
    std::shared_ptr<T> getResourceByName(const std::string& resourceName) const;

    template<typename T>
    std::shared_ptr<T> getResourceByPath(const std::string& resourcePath) const;

    template<typename T>
    std::shared_ptr<T> getResource(const std::function<bool(const std::shared_ptr<Resource>& resourcePtr)>& searchFunc) const;

    // When this method is called, it gives you the resource which has the name of param resourceName.
    // When the resource is not loaded it loads the resource in place.
    // When resource is not found it returns nullptr.
    template<typename T>
    std::shared_ptr<T> getResourceByNameWithOptLoad(const std::string& resourceName);

    template<typename T>
    std::shared_ptr<T> getResourceByPathWithOptLoad(const std::string& path);

    template<typename T>
    std::vector<std::shared_ptr<T>> getResourcesOfType() const;

    private:
    sf::contfree_safe_ptr<std::set<std::shared_ptr<Resource>>> resources;

    std::thread cpuResourceLoaderThread;
    std::thread cpuResourceStatusCheckerThread;
    std::atomic_bool joinLoaderThread = false;

    sf::contfree_safe_ptr<std::deque<ResourceProcessingInfo>> pendingResources;
    sf::contfree_safe_ptr<std::deque<std::shared_future<ResourceProcessingInfo>>> cpuQueue;
    sf::contfree_safe_ptr<std::deque<ResourceProcessingInfo>> gpuQueue;

    template<typename T>
    std::shared_ptr<T> getResourceOptLoad(const std::function<bool(const std::shared_ptr<Resource>& resourcePtr)>& searchFunc);

    void pushToPendingQueue(const ResourceProcessingInfo&& processingInfo);
    std::optional<ResourceProcessingInfo> popFromPendingQueue();

    std::optional<std::shared_future<ResourceProcessingInfo>> popFromCpuQueue();
    void pushToCpuQueue(const std::shared_future<ResourceProcessingInfo>& resourceToLoad);

    void pushToGPUQueue(const ResourceProcessingInfo&& processingInfo);
    std::optional<ResourceProcessingInfo> popFromGPUQueue();

    void processPendingResourcesLoop();
    void findAndAddProperResourcesToPendingQueue();
    std::vector<std::shared_ptr<Resource>> findResourcesToUnload();
    static bool isReadyToUnload(const std::shared_ptr<Resource>& resource);
    std::vector<std::shared_ptr<Resource>> findResourcesToLoad();
    static bool isReadyToLoad(const std::shared_ptr<Resource>& resource);
    void addResourcesToPendingQueue(const std::vector<std::shared_ptr<Resource>>& filteredResources, ProcessingType type);

    void checkCpuResourceStatusLoop();

    void runThreads();
    void joinThreads();

    template<class InputIterator, class Functor>
    [[nodiscard]] std::vector<typename std::iterator_traits<InputIterator>::value_type> static filter(const InputIterator& begin,
                                                                                                      const InputIterator& end, Functor f);
};

template<typename T>
std::vector<ResourceIdentifier> ResourceLibrary::getResourceIdentifiers() const
{
    std::vector<ResourceIdentifier> resourceIdentifiers;
    resourceIdentifiers.reserve(resources->size());

    for(auto resourceIt = resources->begin(); resourceIt != resources->end(); ++resourceIt)
    {
        T* ptr = dynamic_cast<T*>((*resourceIt).get()); 
        if(ptr != nullptr)
        {
            resourceIdentifiers.push_back((*resourceIt)->id);
        }
    }

    return resourceIdentifiers;
}

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResourceByName(const std::string& resourceName) const
{
    const auto findByName = [&resourceName](const std::shared_ptr<Resource>& resourcePtr) {
        const bool validName = resourcePtr->id.getResourceName().string() == resourceName;
        const bool validType = dynamic_cast<T*>(resourcePtr.get()) != nullptr;
        return validName && validType;
    };

    return getResource<T>(findByName);
}

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResourceByPath(const std::string& resourcePath) const
{
    const auto findByPath = [&resourcePath](const std::shared_ptr<Resource>& resourcePtr) {
        const bool validName = resourcePtr->id.getFullPath().string() == resourcePath;
        const bool validType = dynamic_cast<T*>(resourcePtr.get()) != nullptr;
        return validName && validType;
    };

    return getResource<T>(findByPath);
}

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResource(const std::function<bool(const std::shared_ptr<Resource>& resourcePtr)>& searchFunc) const
{
    const auto it = std::find_if(resources->begin(), resources->end(), searchFunc);

    if(it != resources->end())
    {
        const std::shared_ptr<Resource>& ptr = (*it);
        return std::dynamic_pointer_cast<T>(ptr);
    }

    return nullptr;
}

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResourceByNameWithOptLoad(const std::string& resourceName)
{
    const auto compareByTypeAndResourceName = [&resourceName](const std::shared_ptr<Resource>& resourcePtr) {
        const bool validName = resourcePtr->id.getResourceName().string() == resourceName;
        const bool validType = dynamic_cast<T*>(resourcePtr.get()) != nullptr;
        return validName && validType;
    };

    return getResourceOptLoad<T>(compareByTypeAndResourceName);
}

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResourceByPathWithOptLoad(const std::string& path)
{
    const auto compareByTypeAndResourceName = [&path](const std::shared_ptr<Resource>& resourcePtr) {
        const bool validName = resourcePtr->id.getFullPath().string() == path;
        const bool validType = dynamic_cast<T*>(resourcePtr.get()) != nullptr;
        return validName && validType;
    };

    return getResourceOptLoad<T>(compareByTypeAndResourceName);
}

template<typename T>
std::shared_ptr<T> ResourceLibrary::getResourceOptLoad(const std::function<bool(const std::shared_ptr<Resource>& resourcePtr)>& searchFunc)
{
    const auto it = std::find_if(resources->begin(), resources->end(), searchFunc);

    if(it == resources->end())
    {
        return nullptr;
    }

    if((*it)->isResourceReady())
    {
        return std::dynamic_pointer_cast<T>((*it));
    }

    const std::shared_ptr<Resource> resource = (*it);
    const auto gpuResource = dynamic_cast<GPUResource*>(resource.get());

    while(!resource->isLoadedIntoRAM())
        ;  // waiting for loading resource into RAM

    if(gpuResource == nullptr)
    {
        // resource is not of type GPUResource so it can be returned
        return std::dynamic_pointer_cast<T>(resource);
    }

    // when resource is of GPUResource type and it is loaded into RAM it must be inserted to GPU queue
    // we need to look for this resource by searching it in the GpuQueue
    while(true)
    {
        auto readonlyGpuQueue = sf::xlock_safe_ptr(gpuQueue);
        const auto gpuQueueIt =
            std::find_if(readonlyGpuQueue->begin(), readonlyGpuQueue->end(),
                         [&resource](const ResourceProcessingInfo& processingInfo) { return processingInfo.resource.get() == resource.get(); });

        if(gpuQueueIt == readonlyGpuQueue->end())
        {
            continue;
        }

        readonlyGpuQueue->erase(gpuQueueIt);

        gpuResource->gpuLoad();
        resource->endProcessing();
        break;
    }

    return std::dynamic_pointer_cast<T>(resource);
}

template<typename T>
std::vector<std::shared_ptr<T>> ResourceLibrary::getResourcesOfType() const
{
    auto filteredResources = filter(resources->begin(), resources->end(),
                                    [](const std::shared_ptr<Resource>& resource) { return dynamic_cast<T*>(resource.get()) != nullptr; });

    std::vector<std::shared_ptr<T>> resourcesOfTypeT;
    resourcesOfTypeT.reserve(filteredResources.size());

    for(const std::shared_ptr<Resource>& resource : filteredResources)
    {
        resourcesOfTypeT.push_back(std::dynamic_pointer_cast<T>(resource));
    }

    return resourcesOfTypeT;
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