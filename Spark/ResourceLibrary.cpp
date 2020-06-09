#include "ResourceLibrary.h"

#include "Logging.h"
#include "ResourceFactory.h"
#include "Spark.h"

namespace spark::resourceManagement
{
void ResourceLibrary::setup()
{
    runThreads();
}

void ResourceLibrary::cleanup()
{
    joinThreads();
}

void ResourceLibrary::createResources(const std::filesystem::path& pathToResources)
{
    for(const auto& path : std::filesystem::recursive_directory_iterator(pathToResources))
    {
        std::optional<std::shared_ptr<Resource>> optResource = ResourceFactory::createResource(path);
        if(optResource != std::nullopt && optResource.value() != nullptr)
        {
            resources->insert(optResource.value());
        }
    }
}

void ResourceLibrary::processGpuResources()
{
    while(true)
    {
        auto resourceOpt = popFromGPUQueue();

        if(resourceOpt == std::nullopt)
            break;

        ResourceProcessingInfo processingInfo = resourceOpt.value();

        GPUResource* gpuResource = dynamic_cast<GPUResource*>(processingInfo.resource.get());
        if(processingInfo.type == ProcessingType::LOAD)
        {
            if(!gpuResource->gpuLoad())
            {
                SPARK_WARN("GPU resource was not loaded! Pushing it to GPU loading queue againg!");
                pushToGPUQueue({processingInfo.resource, ProcessingType::LOAD, false});
            }
        }
        else if(processingInfo.type == ProcessingType::UNLOAD)
        {
            if(!gpuResource->gpuUnload())
            {
                SPARK_WARN("GPU resource was not unloaded properly! Pushing it to GPU queue againg!");
                pushToGPUQueue({processingInfo.resource, ProcessingType::UNLOAD, false});
            }
        }
        // it means the resource was loaded properly and the processing state is finished
        processingInfo.resource->endProcessing();
    }
}

size_t ResourceLibrary::getLoadedResourcesCount() const
{
    unsigned int counter = 0;
    for(auto resourceIt = resources->begin(); resourceIt != resources->end(); ++resourceIt)
    {
        if((*resourceIt)->isResourceReady())
        {
            ++counter;
        }
    }
    return counter;
}

void ResourceLibrary::pushToPendingQueue(const ResourceProcessingInfo&& processingInfo)
{
    pendingResources->push_back(processingInfo);
}

std::optional<ResourceProcessingInfo> ResourceLibrary::popFromPendingQueue()
{
    if(pendingResources->empty())
        return std::nullopt;

    const auto processingInfo = pendingResources->front();
    pendingResources->pop_front();
    return processingInfo;
}

std::optional<std::shared_future<ResourceProcessingInfo>> ResourceLibrary::popFromCpuQueue()
{
    if(cpuQueue->empty())
        return std::nullopt;

    auto processingInfo = std::move(cpuQueue->front());
    cpuQueue->pop_front();
    return processingInfo;
}

void ResourceLibrary::pushToCpuQueue(const std::shared_future<ResourceProcessingInfo>& resourceToLoad)
{
    cpuQueue->push_back(resourceToLoad);
}

void ResourceLibrary::pushToGPUQueue(const ResourceProcessingInfo&& processingInfo)
{
    gpuQueue->push_back(processingInfo);
}

std::optional<ResourceProcessingInfo> ResourceLibrary::popFromGPUQueue()
{
    if(gpuQueue->empty())
        return std::nullopt;

    auto processingInfo = std::move(gpuQueue->front());
    gpuQueue->pop_front();
    return processingInfo;
}

void ResourceLibrary::processPendingResourcesLoop()
{
    while(true)
    {
        findAndAddProperResourcesToPendingQueue();

        auto processingInfoOpt = popFromPendingQueue();

        if (joinLoaderThread.load() == true && processingInfoOpt == std::nullopt)
        {
           break;
        }

        if(processingInfoOpt == std::nullopt)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        ResourceProcessingInfo& processingInfo = processingInfoOpt.value();

        const auto processOnCpu = [processingInfo]() {
            bool success = false;
            if(processingInfo.type == ProcessingType::LOAD)
            {
                success = processingInfo.resource->load();
            }
            if(processingInfo.type == ProcessingType::UNLOAD)
            {
                success = processingInfo.resource->unload();
            }

            return ResourceProcessingInfo{processingInfo.resource, processingInfo.type, success};
        };

        std::shared_future<ResourceProcessingInfo> future = std::async(std::launch::async, processOnCpu);
        pushToCpuQueue(future);
    }

    if(!pendingResources->empty())
    {
        SPARK_WARN("Joing the thread but the pendingResources queue is not empty!");
    }
}

void ResourceLibrary::findAndAddProperResourcesToPendingQueue()
{
    addResourcesToPendingQueue(findResourcesToUnload(), ProcessingType::UNLOAD);
    addResourcesToPendingQueue(findResourcesToLoad(), ProcessingType::LOAD);
}

std::vector<std::shared_ptr<Resource>> ResourceLibrary::findResourcesToUnload()
{
    return filter(resources->begin(), resources->end(), isReadyToUnload);
}

bool ResourceLibrary::isReadyToUnload(const std::shared_ptr<Resource>& resource)
{
    const bool resourceIsNotUsed = resource.use_count() == 1;
    const bool resourceIsLoaded = resource->isResourceReady();
    const bool resourceIsNotInProcessingState =
        resource->processingFlag() == false;  // it means that resource can be added only if it's not processed at this moment
    return resourceIsNotUsed && resourceIsLoaded && resourceIsNotInProcessingState;
}

std::vector<std::shared_ptr<Resource>> ResourceLibrary::findResourcesToLoad()
{
    return filter(resources->begin(), resources->end(), isReadyToLoad);
}

bool ResourceLibrary::isReadyToLoad(const std::shared_ptr<Resource>& resource)
{
    const bool resourceIsUsed = resource.use_count() > 1;
    const bool resourceIsNotLoaded = !resource->isResourceReady();
    const bool resourceIsNotInProcessingState =
        resource->processingFlag() == false;  // it means that resource can be added only if it's not processed at this moment
    return resourceIsUsed && resourceIsNotLoaded && resourceIsNotInProcessingState;
}

void ResourceLibrary::addResourcesToPendingQueue(const std::vector<std::shared_ptr<Resource>>& filteredResources, ProcessingType type)
{
    for(const auto& resource : filteredResources)
    {
        // const auto gpuResource = dynamic_cast<GPUResource*>(resource.get());
        // if (gpuResource != nullptr)
        //{
        resource->startProcessing();
        pushToPendingQueue({resource, type, false});
        //}
    }
}

void ResourceLibrary::checkCpuResourceStatusLoop()
{
    while(true)
    {
        auto processingProcessingStatusOpt = popFromCpuQueue();

        if (joinLoaderThread.load() == true && processingProcessingStatusOpt == std::nullopt)
        {
           break;
        }
        
        if(processingProcessingStatusOpt == std::nullopt)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        std::shared_future<ResourceProcessingInfo>& resourceStatus = processingProcessingStatusOpt.value();

        if(const std::future_status status = resourceStatus.wait_for(std::chrono::microseconds(10)); status != std::future_status::ready)
        {
            pushToCpuQueue(resourceStatus);
            continue;
        }

        auto& processingInfo = resourceStatus.get();

        if(!processingInfo.success)
        {
            SPARK_WARN("Resource cpu load unsuccesfull! Trying to load again (pushing resource to pending queue)!");
            pushToPendingQueue({processingInfo.resource, processingInfo.type, false});
        }
        else
        {
            GPUResource* gpuResource = dynamic_cast<GPUResource*>(processingInfo.resource.get());
            if(gpuResource)
            {
                pushToGPUQueue({processingInfo.resource, processingInfo.type, false});
            }
            else
            {
                processingInfo.resource->endProcessing();
            }
        }
    }

    if(!cpuQueue->empty())
    {
        SPARK_WARN("Joing the thread but the cpuQueue is not empty!");
    }
}

void ResourceLibrary::runThreads()
{
    joinLoaderThread.store(false);
    cpuResourceLoaderThread = std::thread([this]() { processPendingResourcesLoop(); });
    cpuResourceStatusCheckerThread = std::thread([this]() { checkCpuResourceStatusLoop(); });
}

void ResourceLibrary::joinThreads()
{
    joinLoaderThread.store(true);
    if(cpuResourceLoaderThread.joinable())
        cpuResourceLoaderThread.join();
    if(cpuResourceStatusCheckerThread.joinable())
        cpuResourceStatusCheckerThread.join();
}
}  // namespace spark::resourceManagement
