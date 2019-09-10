#include "ResourceManager.h"
#include "ResourceLoader.h"


Texture ResourceManager::findTexture(const std::string&& path)
{
	for(auto& tex_it : textures)
	{
		if (tex_it.path == path)
			return tex_it;
	}
	return {};
}

Model* ResourceManager::findModel(const std::string&& path)
{
	const auto& it = models.find(path);
	if (it != models.end())
	{
		return it->second;
	}
	return nullptr;
}

ResourceManager::ResourceManager()
{
}


ResourceManager::~ResourceManager()
{
}

ResourceManager* ResourceManager::getInstance()
{
	static ResourceManager* resource_manager = nullptr;
	if(resource_manager == nullptr)
	{
		resource_manager = new ResourceManager();
	}
	return resource_manager;
}

void ResourceManager::loadResources()
{
	textures = ResourceLoader::loadTextures(R"(C:\Studia\Semestr6\SparkRenderer\res)");
	models = ResourceLoader::loadModels(R"(C:\Studia\Semestr6\SparkRenderer\res\models)");
}
