#include "GPUResource.h"

bool spark::resourceManagement::GPUResource::isLoadedIntoDeviceMemory() const
{
	return loadedIntoDeviceMemory.load();
}

void spark::resourceManagement::GPUResource::setLoadedIntoDeviceMemory(bool state)
{
	loadedIntoDeviceMemory.store(state);
}
