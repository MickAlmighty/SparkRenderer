#include "Resource.h"

namespace spark::resourceManagement
{

Resource::Resource(const ResourceIdentifier& identifier) : id(identifier){}

bool Resource::operator<(const Resource& resource) const
{
	return id < resource.id;
}

void Resource::setLoadedIntoRam(bool state)
{
	loadedIntoRAM.store(state);
}

void Resource::startProcessing()
{
	processing.store(true);
}

void Resource::endProcessing()
{
	processing.store(false);
}

bool Resource::processingFlag() const
{
	return processing.load();
}

std::string Resource::getName() const
{
	return id.getResourceName().string();
}

bool Resource::isLoadedIntoRAM() const
{
	return loadedIntoRAM.load();
}
}
