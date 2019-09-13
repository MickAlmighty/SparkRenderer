#include "ResourceManager.h"
#include "ResourceLoader.h"
#include "Spark.h"



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

std::shared_ptr<Shader>& ResourceManager::getShader(ShaderType type)
{
	const auto& it = shaders.find(type);
	if (it != shaders.end())
	{
		return it->second;
	}

	throw std::exception("Cannot find shader! Probably shader was not loaded!");
}

ResourceManager::ResourceManager()
{
}


ResourceManager::~ResourceManager()
{
	for(auto& tex_it: textures)
	{
		glDeleteTextures(1, &tex_it.ID);
	}
	textures.clear();

	for(auto& model_it: models)
	{
		delete model_it.second;
	}
	models.clear();
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
	std::filesystem::path shaderDir = Spark::pathToResources;
	shaderDir.append("shaders\\");
	shaders.emplace(DEFAULT_SHADER, std::make_shared<Shader>(shaderDir.string() + "default.vert", shaderDir.string() + "default.frag"));
	shaders.emplace(SCREEN_SHADER, std::make_shared<Shader>(shaderDir.string() + "screen.vert", shaderDir.string() + "screen.frag"));
	shaders.emplace(POSTPROCESSING_SHADER, std::make_shared<Shader>(shaderDir.string() + "postprocessing.vert", shaderDir.string() + "postprocessing.frag"));

	textures = ResourceLoader::loadTextures(Spark::pathToResources);
	models = ResourceLoader::loadModels(Spark::pathToModels);
}

void ResourceManager::cleanup()
{
	textures.clear();
	
	for (auto model_it : models)
		delete model_it.second;
	models.clear();
	shaders.clear();
}
