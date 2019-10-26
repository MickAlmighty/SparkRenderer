#include "EngineSystems/ResourceManager.h"

#include "Mesh.h"
#include "ResourceLoader.h"
#include "Shader.h"
#include "Spark.h"
#include "Timer.h"

namespace spark {

ResourceManager* ResourceManager::getInstance()
{
	static auto resource_manager = new ResourceManager();
	return resource_manager;
}

void ResourceManager::addTexture(Texture tex)
{
	const auto tex_it = std::find_if(std::begin(textures), std::end(textures), [&tex](const Texture& texture)
	{
		return texture.path == tex.path;
	});
	if (tex_it != std::end(textures))
	{
		textures.erase(tex_it);

	}
	textures.push_back(tex);
}

Texture ResourceManager::findTexture(const std::string&& path) const
{
	for (auto& tex_it : textures)
	{
		if (tex_it.path == path)
			return tex_it;
	}
	return {};
}

std::vector<Mesh> ResourceManager::findModelMeshes(const std::string& path) const
{
	const auto& it = models.find(path);
	if (it != models.end())
	{
		return it->second;
	}
	return {};
}

std::vector<std::string> ResourceManager::getPathsToModels() const
{
	std::vector<std::string> paths;
	for (auto& element : models)
	{
		paths.push_back(element.first);
	}
	return paths;
}

std::shared_ptr<Shader> ResourceManager::getShader(const ShaderType& type) const
{
	const auto& it = shaders.find(type);
	if (it != shaders.end())
	{
		return it->second;
	}

	throw std::exception("Cannot find shader! Probably shader was not loaded!");
}

std::shared_ptr<Shader> ResourceManager::getShader(const std::string& name) const
{
	const auto it = std::find_if(std::begin(shaders), std::end(shaders), [&name](const std::pair<ShaderType, std::shared_ptr<Shader>>& pair)
	{
		return pair.second->name == name;
	});
	if (it != std::end(shaders))
	{
		return it->second;
	}
	return nullptr;
}

std::vector<std::string> ResourceManager::getShaderNames() const
{
	std::vector<std::string> shaderNames;
	shaderNames.reserve(shaders.size());
	for(const auto& [shaderEnum, shader_ptr] : shaders)
	{
		shaderNames.push_back(shader_ptr->name);
	}
	return shaderNames;
}

std::vector<Texture> ResourceManager::getTextures() const
{
	return textures;
}

void ResourceManager::loadResources()
{
	std::filesystem::path shaderDir = Spark::pathToResources;
	shaderDir.append("shaders\\");
	shaders.emplace(ShaderType::DEFAULT_SHADER, std::make_shared<Shader>(shaderDir.string() + "default.glsl"));
	shaders.emplace(ShaderType::SCREEN_SHADER, std::make_shared<Shader>(shaderDir.string() + "screen.glsl"));
	shaders.emplace(ShaderType::POSTPROCESSING_SHADER, std::make_shared<Shader>(shaderDir.string() + "postprocessing.glsl"));
	shaders.emplace(ShaderType::LIGHT_SHADER, std::make_shared<Shader>(shaderDir.string() + "light.glsl"));
	shaders.emplace(ShaderType::MOTION_BLUR_SHADER, std::make_shared<Shader>(shaderDir.string() + "motionBlur.glsl"));

	Timer timer("ResourceManager::loadResources");
	textures = ResourceLoader::loadTextures(Spark::pathToResources);
	models = ResourceLoader::loadModels(Spark::pathToModelMeshes);
}

void ResourceManager::cleanup()
{
	for (auto& tex_it : textures)
	{
		glDeleteTextures(1, &tex_it.ID);
	}
	textures.clear();
	for (auto& model_it : models)
	{
		//std::vector<Mesh> meshes = model_it.second;
		for (Mesh& mesh : model_it.second)
		{
			mesh.cleanup();
		}
	}
	models.clear();
	shaders.clear();
}

}
