#ifndef RESOURCE__H
#define RESOURCE__H

#include <future>

#include "ResourceIdentifier.h"

namespace spark::resourceManagement
{
class Resource abstract
{
    public:
    ResourceIdentifier id;
    
    Resource(const ResourceIdentifier& identifier);
    virtual ~Resource() = default;

    bool operator<(const Resource& resource) const;

    virtual bool load() = 0;
    virtual bool unload() = 0;
    virtual bool isResourceReady() = 0;

    std::string getName() const;
    bool isLoadedIntoRAM() const;

    void startProcessing();
    void endProcessing();
    bool processingFlag() const;

    protected:
    void setLoadedIntoRam(bool state);

    private:
    std::atomic_bool loadedIntoRAM{ false };
    std::atomic_bool processing{ false };
};
}
#endif