#pragma once

#include <atomic>

namespace spark::resourceManagement
{
class GPUResource abstract
{
    public:
    virtual ~GPUResource() = default;

    virtual bool gpuLoad() = 0;
    virtual bool gpuUnload() = 0;
    bool isLoadedIntoDeviceMemory() const;

    protected:
    void setLoadedIntoDeviceMemory(bool state);

    private:
    std::atomic_bool loadedIntoDeviceMemory{false};
};
}