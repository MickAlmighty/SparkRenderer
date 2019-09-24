#include <EngineSystems/ResourceManager.h>
#include <ResourceLoader.h>
#include <Spark.h>
#include <Shader.h>
#include <Mesh.h>

void ResourceManager::addTexture(Texture tex)
{
	auto tex_it = std::find_if(std::begin(textures), std::end(textures), [&tex](const Texture& texture )
	{
		return texture.path == tex.path;
	});
	if(tex_it != std::end(textures))
	{
		textures.erase(tex_it);
		
	}
	textures.push_back(tex);
}

Texture ResourceManager::findTexture(const std::string&& path)
{
	for(auto& tex_it : textures)
	{
		if (tex_it.path == path)
			return tex_it;
	}
	return {};
}

std::vector<Mesh> ResourceManager::findModelMeshes(const std::string& path)
{
	const auto& it = models.find(path);
	if (it != models.end())
	{
		return it->second;
	}
	return {};
}

std::vector<std::string> ResourceManager::getPathsToModels()
{
	std::vector<std::string> paths;
	for (auto& element : models)
	{
		paths.push_back(element.first);
	}
	return paths;
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

std::vector<Texture> ResourceManager::getTextures()
{
	return textures;
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
	shaders.emplace(ShaderType::DEFAULT_SHADER, std::make_shared<Shader>(shaderDir.string() + "default.vert", shaderDir.string() + "default.frag"));
	shaders.emplace(ShaderType::SCREEN_SHADER, std::make_shared<Shader>(shaderDir.string() + "screen.vert", shaderDir.string() + "screen.frag"));
	shaders.emplace(ShaderType::POSTPROCESSING_SHADER, std::make_shared<Shader>(shaderDir.string() + "postprocessing.vert", shaderDir.string() + "postprocessing.frag"));

	textures = ResourceLoader::loadTextures(Spark::pathToResources);
	models = ResourceLoader::loadModels(Spark::pathToModelMeshes);
}

void ResourceManager::cleanup()
{
	for(auto& tex_it : textures)
	{
		glDeleteTextures(1, &tex_it.ID);
	}
	textures.clear();
	for(auto& model_it: models)
	{
		//std::vector<Mesh> meshes = model_it.second;
		for(Mesh& mesh : model_it.second)
		{
			mesh.cleanup();
		}
	}
	models.clear();
	shaders.clear();
}
